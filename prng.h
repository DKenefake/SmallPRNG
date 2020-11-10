#pragma once

#include<inttypes.h>
#include<immintrin.h>
#include<random>

#ifdef __GNUC__
#include<x86intrin.h>
#else
#include<intrin.h>
#endif

#ifdef _MSC_VER
#define _INLINE __forceinline
#elif __GNUC__
#define _INLINE __attribute__((always_inline)) inline
#else
#define _INLINE inline
#endif



namespace smallprng{

#ifdef SMALLPRNG_USE_RDSEED
constexpr auto use_rdseed = true;
constexpr auto use_rddevice = false;
#endif

#ifndef SMALLPRNG_USE_RDSEED
constexpr auto use_rdseed = false;
constexpr auto use_rddevice = true;
#endif


constexpr auto prng_unbiased = false;

static_assert(use_rdseed^ use_rddevice, "Use either rdseed or random device of entropy not both");

//prng state space struct
template<int N>
struct prng_state {
	static_assert(N > 0, "prng_state must have a postive amount of memory, for prng_state<N> N >=1 ");
	union {
		uint16_t i16[N * 2];
		uint32_t i32[N];
		uint64_t i64[N >> 1];
	};
};

//prng template specialization for N = 1, due to zero length i64
template<>
struct prng_state<1> {
	union {
		uint16_t i16[2];
		uint32_t i32[1];
	};
};

//this can be bypassed if a difference source of entropy wants to be used
template<int N>
prng_state<N> create_state() {
	auto state = prng_state<N>();
	if constexpr (use_rdseed == true) {
		//use the x86 machine instruction to gather entropy
		for (int i = 0; i < N;)
			i += _rdseed32_step(&state.i32[i]);
	}
	if constexpr (use_rddevice == true) {
		//use device to gather entropy, this is risky as it is slow
		std::random_device dev;
		for (int i = 0; i < N; i++)
			state.i32[i] = dev();
	}
	return state;
}

template<int N>
prng_state<N> set_state(uint32_t* dat) {
	auto state = prng_state<N>();
	std::copy(dat, dat + N, state.i32);
	return state;
}


template<int N, typename T, T(*F)(prng_state<N>&)>
class prng {
public:
	prng() {
		state = create_state<N>();
	};

	~prng() {};

	T operator()() {
		return F(state);
	};

	_INLINE
	uint64_t rand_64(uint64_t low, uint64_t high) {
		//unbiased 
		if constexpr (prng_unbiased == true) {
			//Debiased Modulo citation:http://www.pcg-random.org/posts/bounded-rands.html
			uint64_t x, r;
			do {
				x = rand_64();
				r = x % high;
			} while (x - r > (-high));
			return r + low;
		}

		if constexpr (prng_unbiased == false) {
			return rand_64() % high + low;
		}
	}

	float randf() {

		uint32_t v = rand_32();
		//transforms v into a [0,1) float 
		union {
			float f;
			uint32_t i;
		}u;
		u.f = 1.0f;
		u.i = u.i | v >> 9;

		return u.f - 1.0f;
	}

	double rand() {
		uint64_t v = rand_64();
		//transforms v into a [0,1) double 
		union {
			double d;
			uint64_t i;
		}u;
		u.i = UINT64_C(0x3FF) << 52 | v >> 12;

		return u.d - 1.0;
	}

	_INLINE
	uint32_t rand_32() {
		uint32_t v;

		if constexpr (sizeof(T) == 8)
			v = uint32_t(operator()() >> 32);
		if constexpr (sizeof(T) == 4)
			v = operator()();
		if constexpr (sizeof(T) == 2)
			v = uint32_t(operator()()) << 16 | uint32_t(operator()());

		return v;
	}

	_INLINE
	uint64_t rand_64() {
		uint64_t v;

		if constexpr (sizeof(T) == 8)
			v = operator()();
		if constexpr (sizeof(T) == 4)
			v = (uint64_t(operator()()) << 32) | uint64_t(operator()());
		if constexpr (sizeof(T) == 2)
			v = uint64_t(operator()()) | (uint64_t(operator()()) << 16) | (uint64_t(operator()()) << 32) | (uint64_t(operator()()) << 48);

		return v;
	}
	T rand_poisson(double l) {
		int n = 0;
		int m = 0;
		int cutoff = 10;

		//poisson(x) = sum(poisson(y_i)) s.t. sum(y_i) = x
		//At most sample from poisson(10) for numerical accuracy of the fp math

		//TODO: implement 

		while (l > cutoff) {
			m += rand_poisson(cutoff);
			l -= cutoff;
		}

		double cdf = rand() * std::exp(l);
		double prod = 1.0;
		double denom = 1.0;
		double sum = 1;

		while (sum < cdf) {
			n++;
			prod *= l;
			denom *= n;
			sum += prod / denom;
		}

		return n + m;
	}

	double rand_normal(double mean = 0, double std = 1) {

		double u, v, s;
		do {
			u = rand() * 2 - 1;
			v = rand() * 2 - 1;
			s = u * u + v * v;
		} while (s >= 1 || s == 0);
		s = std::sqrt(-2.0 * std::log(s) / s);
		auto normal_spare = v * s;
		return normal_spare * std * s + mean;
	}

	double rand(double low, double high) {
		return rand() * (high - low) + low;
	};

	float randf(float low, float high) {
		return randf() * (high - low) + low;
	}

public:
	prng_state<N> state;
};

template<typename prng, int k>
class prng_kd{
public:

	prng_kd() {
		for (int i = 0; i < k; i++)
			prngs[i] = prng();
		counter = 0LL;
	};
	
	~prng_kd() {};

	_INLINE
	float randf() {
		float result = prngs[counter].randf();
		update_counter();
		return result;
	}

	_INLINE
	double rand() {
		double result = prngs[counter].rand();
		update_counter();
		return result;
	}

	_INLINE
	uint32_t rand_32() {
		uint32_t result = prngs[counter].rand_32();
		update_counter();
		return result;
	}

	_INLINE
	uint64_t rand_64() {
		uint64_t result = prngs[counter].rand_64();
		update_counter();
		return result;
	}


	_INLINE
	void update_counter() {
		counter++;
		counter %= k;
	}

private:
	prng prngs[k];
	uint64_t counter;
};


_INLINE
uint32_t middle_square(prng_state<6>& s) {
	uint64_t x = s.i64[0];
	uint64_t w = s.i64[1];
	w += s.i64[2];
	x *= x;
	x += w;
	x = (x >> 32) | (x << 32);
	s.i64[0] = x;
	s.i64[1] = w;
	return (uint32_t)x;
}

_INLINE
uint32_t xorshift32(prng_state<1>& s) {
	uint32_t x = s.i32[0];
	x ^= x << 13;
	x ^= x >> 17;
	x ^= x << 5;
	s.i32[0] = x;
	return x;
}

_INLINE
uint64_t xorshift64(prng_state<2>& s) {
	uint64_t x = s.i64[0];
	x ^= x << 13;
	x ^= x >> 7;
	x ^= x << 17;
	s.i64[0] = x;
	return x;
}

_INLINE
uint32_t xorshift128(prng_state<4>& s) {
	uint32_t t = s.i32[3];
	uint32_t const s_ = s.i32[0];
	s.i32[3] = s.i32[2];
	s.i32[2] = s.i32[1];
	s.i32[1] = s_;

	t ^= t << 11;
	t ^= t >> 8;
	s.i32[0] = t ^ s_ ^ (s_ >> 19);
	return s.i32[0];
}


_INLINE
uint64_t xorshift128plus(prng_state<4>& s) {
	uint64_t s1 = s.i64[0];
	uint64_t s0 = s.i64[1];
	s.i64[0] = s0;
	s1 ^= s1 << 23;
	s1 ^= s1 >> 17;
	s1 ^= s0;
	s1 ^= s0 >> 26;
	s.i64[1] = s1;
	return s1 + s0;
}

template<typename T>
_INLINE
T rot(T x, int k) {
	constexpr uint32_t bit_count = sizeof(T) << 3;
	return (x << k) | (x >> (bit_count - k));
}


_INLINE
uint64_t xoshiro256ss(prng_state<8>& s) {
	uint64_t const result = rot(s.i64[1] * 5, 7) * 9;
	uint64_t const t = s.i64[1] << 17;

	s.i64[2] ^= s.i64[0];
	s.i64[3] ^= s.i64[1];
	s.i64[1] ^= s.i64[2];
	s.i64[0] ^= s.i64[3];

	s.i64[2] ^= t;
	s.i64[3] = rot(s.i64[3], 45);

	return result;
}

_INLINE
uint64_t fortran_lcg(prng_state<2>& s) {
	uint64_t m = 2862933555777941757UL;
	uint64_t a = 1013904243UL;

	auto return_val = s.i64[0];
	s.i64[0] = s.i64[0] * m + a;

	return return_val;
}

_INLINE
uint32_t squares(prng_state<4>& s) {
	// A 2020 modification od middle squares 
	// https://arxiv.org/pdf/2004.06278v2.pdf
	uint64_t x, y, z;
	y = x = s.i64[0] * s.i64[1];
	z = y + s.i64[1];
	x = x * x + y;
	x = (x >> 32) | (x << 32);
	x = x * x + z;
	x = (x >> 32) | (x << 32);
	s.i64[0]++;
	return (x * x + y) >> 32;
}

_INLINE
uint64_t jsf(prng_state<8>& s) {
	//Bob's noncryptographic prng
	// https://burtleburtle.net/bob/rand/smallprng.html
	uint64_t e = s.i64[0] - rot(s.i64[1], 7);
	s.i64[0] = s.i64[1] ^ rot(s.i64[2], 13);
	s.i64[1] = s.i64[2] + rot(s.i64[3], 37);
	s.i64[2] = s.i64[3] + e;
	s.i64[3] = e + s.i64[0];
	return s.i64[3];
}

_INLINE
uint32_t sfc32(prng_state<4>& s){
	//http://wwwlgis.informatik.uni-kl.de/cms/fileadmin/publications/2020/thesis.pdf
	uint32_t t = s.i32[0] + s.i32[1] + s.i32[3]++;
	s.i32[0] = s.i32[1] ^ (s.i32[1] >> 9);
	s.i32[1] = s.i32[2] ^ (s.i32[2] << 3);
	s.i32[2] = ((s.i32[2] << 21) | (s.i32[2] >> (32 - 21))) + t;
	return t;
}

_INLINE
uint32_t splitmix32(prng_state<4>& s){
	//http://wwwlgis.informatik.uni-kl.de/cms/fileadmin/publications/2020/thesis.pdf
	s.i64[1] |= 1;

	uint64_t seed = s.i64[0];
	s.i64[0] += s.i64[1];
	seed ^= seed >> 33;
	seed *= 0x62a9d9ed799705f5;
	seed ^= seed >> 28;
	seed *= 0xcb24d0a5c88c35b3;
	return uint32_t(seed >> 32);
}

_INLINE
uint32_t rdrand(prng_state<1>& s) {
	while (!_rdrand32_step(&s.i32[0]));
	return s.i32[0];
}

_INLINE
uint32_t rdseed(prng_state<1>& s) {
	while (!_rdseed32_step(&s.i32[0]));
	return s.i32[0];
}

template<int N>
uint64_t rand_aes(prng_state<4>& s) {
	// source translated from
	//https://github.com/Computeiful/BiRandom/blob/master/BiRandom.h
	static_assert(N > 0, "Needs a Positive number of AES rounds");
	union {
		__m128i i;
		uint64_t U, L;
	}seed;

	auto nonce = _mm_set1_epi32(s.i32[3]);

	seed.i = _mm_set1_epi64x((int64_t)s.i64[0]);
	
	for (int i = 0; i < N; i++)
		seed.i = _mm_aesenc_si128(seed.i, nonce);

	s.i64[0] = seed.L;
	return s.i64[0];
}


//predefined prng's that are included with the solver
using mid_sqare = prng<6, uint32_t, middle_square>;
using xor32 = prng<1, uint32_t, xorshift32>;
using xor64 = prng<2, uint64_t, xorshift64>;
using xor128 = prng<4, uint32_t, xorshift128>;
using xor128_plus = prng<4, uint64_t, xorshift128plus>;
using xs_superstar = prng<8, uint64_t, xoshiro256ss>;
using knuth_lcg = prng<2, uint64_t, fortran_lcg>;
using improved_squares = prng<4, uint32_t, squares>;
using sfc = prng<4, uint32_t, sfc32>;
using splitmix = prng<4, uint32_t, splitmix32>;
using bob_prng = prng<8, uint64_t, jsf>;
using rd_rand = prng<1, uint32_t, rdrand>;
using rd_seed = prng<1, uint32_t, rdseed>;
using aes_1 = prng<4, uint64_t, rand_aes<1>>;
using aes_2 = prng<4, uint64_t, rand_aes<2>>;
using aes_4 = prng<4, uint64_t, rand_aes<4>>;
using aes_8 = prng<4, uint64_t, rand_aes<8>>;

template<int N>
using  aes_N = prng<4, uint64_t, rand_aes<N>>;

//allows comand line arguemnts to change prng type

#if SMALLPRNG_XOR32
using prng_default = xor32;
#elif SMALLPRNG_XOR64
using prng_default = xor64;
#elif SMALLPRNG_XOR128
using prng_default = xor128;
#elif SMALLPRNG_XOR128_PLUS
using prng_default = xor128_plus;
#elif SMALLPRNG_XS_SUPERSTAR
using prng_default = xs_superstar;
#elif SMALLPRNG_KNUTH_LCG
using prng_default = knuth_lcg;
#elif SMALLPRNG_IMPROVED_SQUARES
using prng_default = improved_squares;
#elif SMALLPRNG_SFC
using prng_default = sfc;
#elif SMALLPRNG_SPLITMIX
using prng_default = splitmix;
#elif SMALLPRNG_JSF
using prng_default = bob_prng;
#elif SMALLPRNG_RDRAND
using prng_default = rd_rand;
#elif SMALLPRNG_RDSEED
using prng_default = rd_seed;
#elif SMALLPRNG_AES1
using prng_default = aes_1;
#elif SMALLPRNG_AES2
using prng_default = aes_2;
#elif SMALLPRNG_AES4
using prng_default = aes_4;
#elif SMALLPRNG_AES8
using prng_default = aes_8;
#else
//use defualt
using prng_default = improved_squares;
#endif


}

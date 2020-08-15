#pragma once

#include<inttypes.h>
#include<immintrin.h>
#include<intrin.h>
#include<random>

constexpr auto prng_unbiased = false;
constexpr auto use_rdseed = true;
constexpr auto use_rddevice = false;

static_assert(use_rdseed ^ use_rddevice, "Use either rdseed or random device of entropy not both");

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


template<int N, typename T, T (*F)(prng_state<N>&)>
class prng{
public:
	prng() { 
		state = create_state<N>(); 
	};
	
	~prng() {};

	T operator()() {
		return F(state);
	};

	template<typename U>
	T operator()(U low, U high) {
		//unbiased 
		if constexpr (prng_unbiased == true) {
			//Debiased Modulo citation:http://www.pcg-random.org/posts/bounded-rands.html
			T x, r;
			do {
				x = operator()();
				r = x % high;
			} while (x - r > (-high));
			return r + low;
		}

		if constexpr (prng_unbiased == false) {
			return operator()() % high + low;
		}
	}

	double rand() {
		uint64_t v;
		//allows variable sized generators to use the same function
		if constexpr (sizeof(T) == 8)
			v = operator()();
		if constexpr (sizeof(T) == 4)
			v = (uint64_t(operator()()) << 32) | uint64_t(operator()());
		if constexpr (sizeof(T) == 2)
			v = uint64_t(operator()()) | (uint64_t(operator()()) << 16) | (uint64_t(operator()()) << 32) | (uint64_t(operator()()) << 48);

		//transforms v into a [0,1) double 
		union{
			double d;
			uint64_t i;
		}u;
		u.i = UINT64_C(0x3FF) << 52 | v >> 12;
			
		return u.d - 1.0;
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
		if (has_normal_spare) {
			has_normal_spare = false;
			return normal_spare * std + mean;
		}
		else {
			double u, v, s;
			do {
				u = rand() * 2 - 1;
				v = rand() * 2 - 1;
				s = u * u + v * v;
			} while (s >= 1 || s== 0);
			s = std::sqrt(-2.0 * std::log(s) / s);
			normal_spare = v * s;
			has_normal_spare = true;
			return normal_spare * std * s + mean;
		}
	}

	double rand(double low, double high) {
		return rand() * (high - low) + low;
	};

private:
	prng_state<N> state;

	double normal_spare = 0;
	bool   has_normal_spare = false;
};

_inline
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

_inline
uint32_t xorshift32(prng_state<1>& s) {
	uint32_t x = s.i32[0];
	x ^= x << 13;
	x ^= x >> 17;
	x ^= x << 5;
	s.i32[0] = x;
	return x;
}

_inline
uint64_t xorshift64(prng_state<2>& s) {
	uint64_t x = s.i64[0];
	x ^= x << 13;
	x ^= x >> 7;
	x ^= x << 17;
	s.i32[0] = x;
	return x;
}

_inline
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


_inline
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

uint64_t rol64(uint64_t x, int k) {
	return (x << k) | (x >> (64 - k));
}

_inline
uint64_t xoshiro256ss(prng_state<8>& s) {
	uint64_t const result = rol64(s.i64[1] * 5, 7) * 9;
	uint64_t const t = s.i64[1] << 17;

	s.i64[2] ^= s.i64[0];
	s.i64[3] ^= s.i64[1];
	s.i64[1] ^= s.i64[2];
	s.i64[0] ^= s.i64[3];

	s.i64[2] ^= t;
	s.i64[3] = rol64(s.i64[3], 45);
	
	return result;
}

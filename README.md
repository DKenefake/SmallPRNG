# SmallPRNG
A small header only library for prng implementations using templates. This is a fast library, as the swappable prngs implementations are defined at compile time there is not overhead at program runtime. The xorshift128+ implementation included generates ~1 sample per ns on a 4.5Ghz 8600k. In addition there are k equidimensional generators that can be build from any of the prng impementations. 

# Example of use

In the case that the included prng implementations are not suitable for your application, you can inject your own prng into the library with no performance penalty.

We are going to show an example of making a prng based on xorshift64. 

Included in the library is a templated struct that acts as the state for all prng implementations.

```C++
auto s = prng_state<N>()
```
N is the number of 32 bit blocks of memory the state has. It is accessible in 16, 32, and 64 bit blocks in the following manner

```C++
auto a = s.i16[0] //a is the first 16 bits of memory in the state
auto b = s.i32[0] //b is the first 32 bits of memory in the state
auto c = s.i64[0] //c is the first 64 bits of memory in the state
```

In general, the injected prng function should be of the form

```C++

_inline
INT_TYPE PRNG_NAME(prng_state<N>& state){
  //some operations on the state and generate return value
  return sample;
}

```

To generate the sampler based on that prng is done as follows
```
typedef prng<N, INT_TYPE, PRNG_NAME> my_prng;

auto prng = my_prng()
```

To generate a M equidistributed prng is done as follows
```
typedef prng<N, INT_TYPE, PRNG_NAME> my_prng;

auto prng = prng_kd<prng, M>();

```



Now for a concrete example

```C++
_inline
uint64_t xorshift64(prng_state<2>& s) {
	uint64_t x = s.i64[0];
	x ^= x << 13;
	x ^= x >> 7;
	x ^= x << 17;
	s.i64[0] = x;
	return x;
}
```

The type definition of the custom prng
```C++
typedef prng<2, uint64_t, xorshift64> my_prng;

auto prng = my_prng();
auto prng_10dim = prng_kd<my_prng, 10>();
```
With that done, all of the implemented functions can directly use the given generator
```C++

auto a = prng.rand(); // returns a random double in range of (0,1]

auto b = prng.randf(); //returns a random float in range of (0,1]

auto c = prng.rand_64(); // returns a random uint64_t

auto d = prng.rand_32(); // returns a random uint32_t

auto e = prng.rand_64(0,5); // returns a random uint64_t in range of [0,5)

auto f = prng.rand(2, 5); // returns a random double in range of (2,5]

auto g = prng.randf(2,5); // returns a random float in range of (2, 5]

auto h = prng.rand_poisson(4.3); // returns a sample from a poisson distribution of λ=4.3

auto i = prng.rand_normal(); // returns a normalily distributed sample with with mean = 0 and std = 1

auto g = prng.rand_normal(1.0, 4.5);  // returns a normalily distributed sample with with mean = 1.0 and std = 4.5

```

# Included (P)RNG Algorithms

* Middle Square

* Xorshift32/64/128/128+

* Xoshiro256**

* Knuth's LCG

* Squares

* JSF

* Salsa20

* rdrand/rdseed

* AES PRNG

# SmallPRNG
A small header only library for pseudo random number generators (prngs). This is an extendable library with minimal overhead. It comes with a list of common prng algorithms as well as an easy and intuitive interface to implement your own. The motivation for this library to to test how different prng algorithms effect monte carlo code results but it is an effective general purpose library.

# Projects using SmallPRNG

* [MinSSE](https://github.com/Chronum94/MinSSEMinSSE) - A minimal 2/3D Heisenberg lattice simulator using the stochastic series expansion. 


# Example of use 

Here we will use an AES based prng to generate random numbers. 

```cpp
#include "prng.h"

// makes a prng based on 4 rounds of AES
auto rng = smallprng::aes_4();

double a = rng.rand();   // returns a random double in range (0,1]

float b = prng.randf();  // returns a random float in range (0,1] 

unsigned long long c = prng.rand_64(); // returns a random uint64_t

unsigned int d = prng.rand_32(); // returns a random uint32_t

double e = prng.rand(2, 5); // returns a random double in range of (2,5]

float f = prng.randf(2,5); // returns a random float in range of (2, 5]

double g = prng.rand_poisson(4.3); // returns a sample from a poisson distribution of λ=4.3

double h = prng.rand_normal(); // returns a normalily distributed sample with with mean = 0 and std = 1

double i = prng.rand_normal(1.0, 4.5);  // returns a normalily distributed sample with with mean = 1.0 and std = 4.5
```

A default generator is also supplied if selecting a specific algorithm is not important to your workload. This is a fast generator that gives high quality random numbers.

```
auto rng = smallprng::prng_default();
```

If K dimensional equidistribution is needed, the class ``prng_kd`` is included. This is not generally needed for most uses, except for monte carlo codes ect.

Here is an example of a 10 dimensional eqidistributed prng based on the above algorithm.
```
using smallprng;
auto rng = prng_kd<aes_4, 10>();
```

# List of Included PRNGs and how to access them


* Xorshift32
```C++
auto rng = smallprng::xor32();
```
* Xorshift64
```C++
auto rng = smallprng::xor64();
```
* Xorshift128
```C++
auto rng = smallprng::xor128();
```
* Xorshift128+
```C++
auto rng = smallprng::xor128_plus();
```
* Xoroshiro**
```C++
auto rng = smallprng::xs_superstar();
```
* LCG based on Knuth's Parameters
```C++
auto rng = smallprng::knuth_lcg();
```
* Squared Algorithm from 
```C++
auto rng = smallprng::improved_squares();
```
* SFC32
```C++
auto rng = smallprng::sfc();
```
* SplitMix32
```C++
auto rng = smallprng::splitmix();
```
* JSF
```C++
auto rng = smallprng::bob_prng();
```
* x86 RD Rand
```C++
auto rng = smallprng::rd_rand();
```
* x86 RD Seed
```C++
auto rng = smallprng::rd_seed();
```
* x86 AES 1 Round Chipher
```C++
auto rng = smallprng::aes_1();
```
* x86 AES 2 Round Chipher
```C++
auto rng = smallprng::aes_2();
```
* x86 AES 4 Round Chipher
```C++
auto rng = smallprng::aes_4();
```
* x86 AES 8 Round Chipher
```C++
auto rng = smallprng::aes_8();
```
* x86 AES N Round Chipher
```C++
auto rng = smallprng::aes_N<N>();
```


# Example of Implementing a PRNG

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
using my_prng = prng<N, INT_TYPE, PRNG_NAME>;

auto prng = my_prng()
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
using my_prng = prng<2, uint64_t, xorshift64>;

auto rng = my_prng();
```

To create a K equidimensional prng from a prng algorithm all that is needed is the following. In this example K = 10.

```C++
using my_kd_prng = prng_kd<my_prng, 10>

auto rng = my_kd_prng();
```

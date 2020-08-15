# SmallPRNG
A small header only library for swapping prng implementations using templates. This is a fast library, as the swappable prngs implementations are defined at compile time there is not overhead at program runtime.

# Example of use

In the case that the included prng implementations are not suitble for your application, you can inject your own prng into the library.

Here we are going to show an example of making a prng based on xorshif64. Included in the library is a tamplated struct that acts as the state for all prng implementations.

```C++
auto s = prng_state<N>()
```
N is the number of 32 bit blocks of memory the state has. It is accessible in 16, 32, and 64 bit blocks in the following manner

```C++
auto a = s.i16[0] //a is the first 16 bits of memory in the state
auto b = s.i32[0] //b is the first 32 bits of memory in the state
auto c = s.i64[0] //c is the first 64 bits of memory in the state
```

In general the injected prng function should be of the form

```C++

_inline
INT_TYPE PRNG_NAME(prng_state<N>& state){
  //some operations on the state and generate return type
  return sample;
}

```

And to generate the sampler based on that prng is done as follows
```
typedef prng<N, INT_TYPE, PRNG_NAME> my_prng;

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
typedef prng<2, uint64_t, xorshift64> my_prng;

auto prng = my_prng();
```
With that done, all of the implemented functions can directly use the given generator
```C++
auto b = prng(); // returns a random uint64_t

auto c = prng(0,5); // returns a random uint64_t in range of [0,5)

auto d = prng.rand(); // returns a random double in range of (0,1]

auto e = prng.rand(2, 5); // returns a random double in range of (2,5]

auto f = prng.rand_poisson(4.3); // returns a sample from a poisson distribution of λ=4.3

auto g = prng.rand_normal(); // returns a normalily distributed sample with with mean = 0 and std = 1

auto h = prng.rand_normal(1.0, 4.5);  // returns a normalily distributed sample with with mean = 1.0 and std = 4.5

```

# Benchmark

I have a simple bemchmark here to get relative speeds of each prng algorithm. Eventually I want to run this on different hardware to show relative and absolute performance numebrs on x86 desktop cpus, ARM, ect. The benchmark is summing 1 billion uniformly generated double floating point precission numbers.

I am including the benchmark.cpp file here so that you can run it yourself if you want. If your processor is not listed and you want to share the results feel free to submit a PR so we can add it to this page.

## Results

|    Algorithm    |  Time (mSec) |
|:---------------:|:------------:|
|  Middle Square  |     2286     |
|    Xorshift32   |     2947     |
|    Xorshift64   |     1582     |
|   Xorshift128   |     2460     |
|   Xorshift128+  |     1445     |
|   XoShiro256**  |     5566     |
|   Knuth's LCG   |      948     |
| Improved Square |     2295     |
|    Bob's PRNG   |      984     |
|     Salsa20     |     14853    |

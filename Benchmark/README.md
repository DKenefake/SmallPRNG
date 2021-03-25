# Benchmark

I have a simple bemchmark here to get relative speeds of each prng algorithm. Eventually I want to run this on different hardware to show relative and absolute performance numebrs on x86 desktop cpus, ARM, ect. The benchmark is summing 1 billion uniformly generated double floating point precission numbers. The summary table is time to generate 1 random sample, in units of nanoseconds.

I am including the benchmark.cpp file here so that you can run it yourself if you want. If your processor is not listed and you want to share the results feel free to submit a PR so we can add it to this page.

## Results

|    Algorithm    |     8600k    |     2500k    |
|:---------------:|:------------:|:------------:|
|  Middle Square  |     2.28     |     3.78     |
|    Xorshift32   |     2.94     |     5.97     |
|    Xorshift64   |     1.58     |     3.05     |
|   Xorshift128   |     2.86     |     3.77     |
|   Xorshift128+  |     1.34     |     2.58     |
|   XoShiro256**  |     5.56     |     1.98     |
|   Knuth's LCG   |      .94     |     1.40     |
| Improved Square |     2.40     |     4.43     |
|    Bob's PRNG   |     1.12     |     2.56     |
|   AES 1 Round   |     2.07     |     3.96     |
|   AES 2 Round   |     4.10     |     6.34     |
|   AES 4 Round   |     5.88     |    11.06     |
|   AES 8 Round   |     9.46     |    20.85     |

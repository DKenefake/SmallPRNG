# Benchmark

I have a simple bemchmark here to get relative speeds of each prng algorithm. Eventually I want to run this on different hardware to show relative and absolute performance numebrs on x86 desktop cpus, ARM, ect. The benchmark is summing 1 billion uniformly generated double floating point precission numbers. The summary table is time to generate 1 random sample, in units of nanoseconds.

I am including the benchmark.cpp file here so that you can run it yourself if you want. If your processor is not listed and you want to share the results feel free to submit a PR so we can add it to this page.

## Results

|    Algorithm    |    12700k    |     8600k    |     4970      |     2500k    |
|:---------------:|:------------:|:------------:|:-------------:|:------------:|
|  Middle Square  |     2.09     |     2.28     |     2.96      |     3.78     |
|    Xorshift32   |     2.66     |     2.94     |     3.59      |     5.97     |
|    Xorshift64   |     1.41     |     1.58     |     1.94      |     3.05     |
|   Xorshift128   |     2.33     |     2.86     |     2.99      |     3.77     |
|   Xorshift128+  |     1.15     |     1.34     |     1.63      |     2.58     |
|   XoShiro256**  |     5.78     |     5.56     |     6.17      |     1.98     |
|   Knuth's LCG   |     0.84     |      .94     |     1.18      |     1.40     |
| Improved Square |     1.53     |     2.40     |     2.77      |     4.43     |
|    Bob's PRNG   |     0.85     |     1.12     |     1.28      |     2.56     |
|   AES 1 Round   |     2.01     |     2.07     |     2.72      |     3.96     |
|   AES 2 Round   |     3.81     |     4.10     |     4.57      |     6.34     |
|   AES 4 Round   |     5.10     |     5.88     |     8.33      |    11.06     |
|   AES 8 Round   |     7.56     |     9.46     |    15.86      |    20.85     |

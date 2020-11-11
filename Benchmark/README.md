# Benchmark

I have a simple bemchmark here to get relative speeds of each prng algorithm. Eventually I want to run this on different hardware to show relative and absolute performance numebrs on x86 desktop cpus, ARM, ect. The benchmark is summing 1 billion uniformly generated double floating point precission numbers.

I am including the benchmark.cpp file here so that you can run it yourself if you want. If your processor is not listed and you want to share the results feel free to submit a PR so we can add it to this page.

## Results

|    Algorithm    |     8600k    |     2500k    |
|:---------------:|:------------:|:------------:|
|  Middle Square  |     2286     |     3786     |
|    Xorshift32   |     2947     |     5976     |
|    Xorshift64   |     1582     |     3050     |
|   Xorshift128   |     2860     |     3771     |
|   Xorshift128+  |     1345     |     2580     |
|   XoShiro256**  |     5566     |     1984     |
|   Knuth's LCG   |      948     |     1402     |
| Improved Square |     2404     |     4436     |
|    Bob's PRNG   |     1128     |     2564     |
|   AES 1 Round   |     2071     |     3961     |
|   AES 2 Round   |     4102     |     6344     |
|   AES 4 Round   |     5880     |    11064     |
|   AES 8 Round   |     9462     |    20852     |

# TestU01 - Crush 

Crush is a suite that tests the statistical quality of pseudo random generator algorithms. There are 3 differnt levels of test, 'small crush', 'medium crush' and 'big crush'. I am testing the 32 bit outputs of these generators on a random seed, as well as the bit reversed outputs. I will summarize these in a table as well as include the program outputs in this directory so that you can see what tests were failed.

The number in front of F indicates the number of tests that have failed, and the number in front of S refers to the number of improbable results but not near impossible results. If nothing is marked, then it passes all tests without suspicious results.

These are all run on using different seeds.


## Nonreversed PRNG Results

|     Algorithm    | Small | Medium |   Big  |
|:----------------:|:-----:|:------:|:------:|
|   Middle Square  |       |        |   1S   |
|    Xorshift32    | 6F 1S | 53F 1S | 58F 2S |
|    Xorshift64    |   1F  |   9F   |   8F   |
|    Xorshift128   |       |   6F   |  6F 1S |
|   Xorshift128+   |       |        |   1S   |
|   XoShiro256**   |       |        |        |
|    Knuth's LCG   |       |        |   4F   |
| Improved Squares |       |   1S   |   1S   |
|       SFC32      |       |        |        |
|    SplitMix32    |       |        |        |
|        JSF       |       |        |   2S   |
|    AES 1 Round   |   1S  |  3F 4S |   2F   |
|    AES 2 Round   |       |        |        |
|    AES 4 Round   |       |        |   2S   |
|    AES 8 Round   |       |        |        |

## Reversed PRNG Results
|     Algorithm    | Small | Medium |   Big  |
|:----------------:|:-----:|:------:|:------:|
|   Middle Square  |       |   2S   |        |
|    Xorshift32    |   7F  | 58F 6S |   56F  |
|    Xorshift64    |   1F  |   9F   |   7F   |
|    Xorshift128   |   1F  | 10F 1S | 14F 1S |
|   Xorshift128+   |       |        |        |
|   XoShiro256**   |       |        |   1S   |
|    Knuth's LCG   |       |  4F 1S |   8F   |
| Improved Squares |       |        |        |
|       SFC32      |       |        |        |
|    SplitMix32    |       |        |        |
|        JSF       |       |        |        |
|    AES 1 Round   |       |   1F   |  1F 1S |
|    AES 2 Round   |       |        |        |
|    AES 4 Round   |       |   3S   |        |
|    AES 8 Round   |       |        |        |

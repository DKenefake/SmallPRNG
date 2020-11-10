#include"prng.h"
#include<iostream>
#include<chrono>

//main benchmarking function
template<typename prng>
int benchmark() {
	
	auto start = std::chrono::high_resolution_clock::now();
	
	double sum = 0LL;

	auto my_prng = prng();

	for (uint64_t i = 0; i < 1000000000LL; i++)
		sum += my_prng.rand();

	auto end = std::chrono::high_resolution_clock::now();
	auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);


	std::cout << sum << "        ";

	return diff.count();
}

using namespace smallprng;


int main() {

	std::cout.precision(3);
	std::cout << "Algorithm        Result       Time (mSec)" << std::endl;
	std::cout << "Middle Square    " << benchmark<mid_sqare>() << std::endl;
	std::cout << "Xorshift32       " << benchmark<xor32>() << std::endl;
	std::cout << "Xorshift64       " << benchmark<xor64>() << std::endl;
	std::cout << "Xorshift128      " << benchmark<xor128>() << std::endl;
	std::cout << "Xorshift128+     " << benchmark<xor128_plus>() << std::endl;
	std::cout << "Xoshiro256**     " << benchmark<xs_superstar>() << std::endl;
	std::cout << "Knuth's LCG      " << benchmark<knuth_lcg>() << std::endl;
	std::cout << "Improved Square  " << benchmark<improved_squares>() << std::endl;
	std::cout << "JSF              " << benchmark<bob_prng>() << std::endl;
	std::cout << "AES 1 Round      " << benchmark<aes_1>() << std::endl;
	std::cout << "AES 2 Round      " << benchmark<aes_2>() << std::endl;
	std::cout << "AES 4 Round      " << benchmark<aes_4>() << std::endl;
	std::cout << "AES 8 Round      " << benchmark<aes_8>() << std::endl;

	std::getchar();

	return 1;
}

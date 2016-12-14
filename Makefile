matmult.exe: matmult.cpp
	g++ -o matmult.exe -fopenmp -Wall -O3 -std=c++11 matmult.cpp

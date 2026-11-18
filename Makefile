CXX      ?= g++
CXXFLAGS ?= -O3 -g -std=c++17 -fopenmp

barenn: barenn.cpp
	$(CXX) $(CXXFLAGS) barenn.cpp -o barenn

check: barenn
	./check.sh ./barenn golden.txt
	./check.sh ./barenn golden_lote32.txt optdigits.tra optdigits.tes 32

bench: barenn
	./bench.sh

NVCC      ?= nvcc
NVCCFLAGS ?= -O3 -std=c++17 -arch=sm_86 -lineinfo

barenn_cuda: barenn.cu barenn.cpp
	$(NVCC) $(NVCCFLAGS) barenn.cu -o barenn_cuda

clean:
	rm -f barenn barenn_cuda

.PHONY: check bench clean

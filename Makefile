CXX      ?= g++
CXXFLAGS ?= -O3 -g -std=c++17

barenn: barenn.cpp
	$(CXX) $(CXXFLAGS) barenn.cpp -o barenn

check: barenn
	./check.sh ./barenn golden.txt
	./check.sh ./barenn golden_lote32.txt optdigits.tra optdigits.tes 32

bench: barenn
	./bench.sh

clean:
	rm -f barenn

.PHONY: check bench clean

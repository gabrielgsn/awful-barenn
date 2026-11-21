CXX      ?= g++
CXXFLAGS ?= -O2 -g -std=c++17

barenn: barenn.cpp
	$(CXX) $(CXXFLAGS) barenn.cpp -o barenn

check: barenn
	./check.sh

bench: barenn
	./bench.sh

clean:
	rm -f barenn

.PHONY: check bench clean

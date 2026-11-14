CXX      ?= g++
# C++14: under C++17 the global `beta` clashes with std::beta (via using namespace std)
CXXFLAGS ?= -std=c++14 -O2 -Wall
LDLIBS   ?= -larmadillo

.PHONY: all test clean

all: ctqmc_1d

ctqmc_1d: ctqmc_1d.cpp ctqmc_1d.h
	$(CXX) $(CXXFLAGS) -o $@ ctqmc_1d.cpp $(LDLIBS)

tests/test_main: tests/test_main.cpp ctqmc_1d.cpp ctqmc_1d.h
	$(CXX) $(CXXFLAGS) -DCTQMC_NO_MAIN -o $@ tests/test_main.cpp ctqmc_1d.cpp $(LDLIBS)

test: tests/test_main
	./tests/test_main

clean:
	rm -f ctqmc_1d tests/test_main

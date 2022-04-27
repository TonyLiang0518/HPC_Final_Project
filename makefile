CXX = g++
CXXFLAGS = -std=c++11 -O3 -march=native
LIBS = -L./temp/lib/ -lfftw3 -I./temp/include/ -lm -fopenmp
# turning off auto-vectorization since this can make hand-vectorized code slower
CXXFLAGS += -fno-tree-vectorize



TARGETS = $(basename $(wildcard *.cpp))

all : $(TARGETS)

%:%.cpp *.h
	$(CXX) $(CXXFLAGS) $< $(LIBS) -o $@

clean:
	-$(RM) $(TARGETS) *~

.PHONY: all, clean
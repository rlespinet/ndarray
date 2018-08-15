CXX=g++

MKL_INC=-DMKL_ILP64 -m64 -I${MKLROOT}/include
MKL_LIB=-L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl

CPPFLAGS=-std=c++17 -Ieigen -Ilibsimdpp $(MKL_INC) -O3 -DNDEBUG
LDFLAGS=$(MKL_LIB)

FILES=test.cpp
EXE=test

all: $(EXE)

$(EXE): test.o
	$(CXX) $(LDFLAGS) $^ -o $@

%.o: %.c
	$(CXX) -c $(CPPFLAGS) $^ -o $@

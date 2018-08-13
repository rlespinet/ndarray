CXX=g++
CPPFLAGS=-std=c++17 -Ieigen -Ilibsimdpp -O3 -DNDEBUG

FILES=test.cpp
EXE=test

all: $(EXE)

$(EXE): $(FILES)
	$(CXX) $(CPPFLAGS) $^ -o $@

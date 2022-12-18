CFLAGS := -std=c++14 -lpthread

CFLAGS += -O2 -fopenmp

SOURCES := ./*.cpp
HEADERS := ./*.h

TARGETBIN := openmp-alg

.SUFFIXES:
.PHONY: all clean

all: $(TARGETBIN)

$(TARGETBIN): $(SOURCES) $(HEADERS)
	$(CXX) -o $@ $(CFLAGS) $(SOURCES)

format:
	clang-format -i ./*.cpp ./*.h

clean:
	rm -rf ./genetic-alg
CFLAGS := -std=c++14 -lpthread

CFLAGS += -g

SOURCES := ./*.cpp
HEADERS := ./*.h

TARGETBIN := genetic-alg

.SUFFIXES:
.PHONY: all clean

all: $(TARGETBIN)

$(TARGETBIN): $(SOURCES) $(HEADERS)
	$(CXX) -o $@ $(CFLAGS) $(SOURCES)

format:
	clang-format -i ./*.cpp ./*.h

clean:
	rm -rf ./genetic-alg
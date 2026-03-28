.PHONY: all
all: test

test: test5.cpp ../../utils/mnist_loader.cpp
	g++ -O2 -I../../include -I../../utils test5.cpp ../../utils/mnist_loader.cpp -o test5.o

.PHONY: clean 
clean:
	rm -f test5.o

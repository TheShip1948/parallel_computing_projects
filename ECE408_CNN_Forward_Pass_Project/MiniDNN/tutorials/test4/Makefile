.PHONY: all
all: test

test: test4.cpp ../../utils/mnist_loader.cpp
	g++ -O2 -I../../include -I../../utils test4.cpp ../../utils/mnist_loader.cpp -o test4.o

.PHONY: clean 
clean:
	rm -f test4.o

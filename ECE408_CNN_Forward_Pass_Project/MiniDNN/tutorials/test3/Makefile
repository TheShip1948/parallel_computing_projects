.PHONY: all
all: test
# This rule tells make how to build hello from test3.cpp
test: test3.cpp ../../utils/mnist_loader.cpp
	g++ -O2 -I../../include -I../../utils test3.cpp ../../utils/mnist_loader.cpp -o test3.o

# This rule tells make to delete the program
.PHONY: clean 
clean:
	rm -f test3.o



.PHONY: all
all: test
# This rule tells make how to build hello from test3.cpp
test: test3.cpp
	g++ -O2 -I../../include test3.cpp -o test3.o

# This rule tells make to delete the program
.PHONY: clean 
clean:
	rm -f test3.o



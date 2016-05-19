CC = icc

FLAGS = -mmic -openmp -O3 -ansi-alias -funroll-loops -fomit-frame-pointer -vec-report5 -opt-streaming-stores=always -no-prec-div -no-prec-sqrt -fp-model fast=2 -fimf-precision=low -fimf-domain-exclusion=15

mmap: mmap.o
	$(CC) $(FLAGS) -o mmap mmap.o
mmap.o: mmap.cpp
	$(CC) $(FLAGS) -c -o mmap.o mmap.cpp
        

clean:
	rm -f *.o mmap


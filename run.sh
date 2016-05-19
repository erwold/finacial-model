#!/bin/bash                                                                 

file=mmap

make clean;
make;
ssh mic0 "rm $file 2> /dev/null"
scp ./$file mic0:;
ssh mic0 ./$file;





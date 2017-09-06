## Prefix 1D Array Scan using Thrust

Thrust is a Standard Template Library for CUDA that contains a Collection of data parallel primitives (eg. vectors) and implementations (eg. Sort, Scan, saxpy) that can be used in writing high performance CUDA code.

This folder contains the implementation of inclusive scan on a 1D array using thrust.

#### Running the code

```sh
$ g++ g++ dataset_generator.cpp
$ ./a.out
$ nvcc template.cu
$ ./a.out <input filename> <output filename>
```

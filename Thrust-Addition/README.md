## Addition of 2 arrays using Thrust

Thrust is a Standard Template Library for CUDA that contains a Collection of data parallel primitives (eg. vectors) and implementations (eg. Sort, Scan, saxpy) that can be used in writing high performance CUDA code.

This folder contains the implementation of addition of 2 vectors using Thrust.

#### Running the code

```sh
$ g++ dataset_generator.cpp
$ ./a.out
$ g++ template.cu
$ ./a.out <output_vector> <input_vector1> <input_vector2>
```

## Histogram for an array of numbers

An efficient histogram algorithm for an input array of integers within a given range. Each integer will map into a single bin, so the values will range from 0 to (NUM_BINS - 1). The histogram bins will use unsigned 32-bit counters that are saturated at 127 (i.e. no roll back to 0 allowed).

#### Running the code

```sh
$ g++ dataset_generator.cpp
$ ./a.out
$ nvcc template.cu
$ ./a.out <output_file> <input_file>
```

## Histogram for an array of ASCII values

An efficient histogram algorithm for an input array of ASCII characters. There are 128 ASCII characters and each character will map into its own bin for a fixed total of 128 bins. The histogram bins will be unsigned 32-bit counters that do not saturate. The approach followed was creating a privatized histogram in shared memory for each thread block, then atomically modifying the global histogram.

#### Running the code

```sh
$ g++ dataset_generator.cpp
$ ./a.out
$ nvcc template.cu
$ ./a.out <output_file> <input_file>
```

## GPU Device Queries

#### Running the code

```sh
$ nvcc deviceQuery.cu
$ ./a.out
```

The information obtained from the queries are:
* Device name
* Number of multiprocessors on GPU
* Memory Clock Rate (KHz)
* Memory Bus Width (bits)
* Peak Memory Bandwidth (GB/s)
* Max number of blocks
* Amount of shared memory per block (bytes)
* Amount of global memory (bytes)
* Warp size of GPU (number of threads)
* Amount of constant memory (bytes)

## 7 point stencil

A shared memory based 3 dimensional tiled approach is followed while implementing the 7 point stencil.

#### Running the code

```sh
$ g++ dataset_generator.cpp
$ ./a.out
$ nvcc template.cu
$ ./a.out -i <input ppm filename> <output ppm filename>
```

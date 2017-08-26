## Image Convolution

The code implements an image convolution using a 5 x 5 mask with the tiled shared memory approach. Convolution is used in many fields, such as image processing for image filtering.  

#### Running the code

```sh
$  g++ dataset_generator.cpp
$ ./a.out
$ nvcc template.cu
$ ./a.out <output(ppm)_filename> <input(ppm)_filename> <input_mask(raw)_filename>
```

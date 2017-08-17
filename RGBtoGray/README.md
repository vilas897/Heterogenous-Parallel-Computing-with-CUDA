## Converting an RGB image to a gray scale input image

The input image consists of RGB value triples that needs to be converted to a single gray scale image pixel value using the formula:

*** gray = 0.21*r + 0.71*g + 0.07*b ***

#### Running the code

```sh
$ g++ dataset_generator.cpp
$ ./a.out
$ nvcc template.cu
$ ./a.out <output_image> <input_image>
```

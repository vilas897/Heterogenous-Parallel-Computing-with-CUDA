## Gaussian Blur for images

An image is represented as `RGB float` values. The code operates directly on the RGB float values and uses a 3x3 Box Filter to blur the original image to produce the blurred image (Gaussian Blur).

#### Running the code

```sh
$ g++ dataset_generator.cpp
$ ./a.out
$ g++ template.cu
$ ./a.out <output_image> <input_image>
```

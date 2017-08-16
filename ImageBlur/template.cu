#include<cuda.h>
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include "wb.h"

#define BLUR_SIZE 1
#define isValid(X, Y) (X >= 0 && Y>=0 && X < height && Y < width)

//@@ INSERT CODE HERE
__global__ void blur(float* input, float* output, int height, int width)
{
    int X = blockIdx.x * blockDim.x + threadIdx.x;
    int Y = blockIdx.y * blockDim.y + threadIdx.y;

    float ans1=0, ans2 = 0, ans3 = 0;
    int count=0;
    for(int i =  X - 1; i <= X+1; i++)
      for(int j = Y - 1; j<= Y+1; j++)
      {
        if(isValid(i, j))
          { ans1 += input[3 * (i*width + j)];
            ans2 += input[3 * (i*width + j) + 1];
            ans3 += input[3 * (i*width + j) + 2];
            count++;
          }
      }

    ans1 = ans1/ count;
    ans2 = ans2/ count;
    ans3 = ans3/ count;

    output[ 3 * (X*width + Y)] = ans1;
    output[ 3 * (X*width + Y) + 1] = ans2;
    output[ 3 * (X*width + Y) + 2] = ans3;
}

int main(int argc, char *argv[]) {

  int imageWidth;
  int imageHeight;
  char *inputImageFile;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  float *deviceInputImageData;
  float *deviceOutputImageData;


  /* parse the input arguments */
  //@@ Insert code here
  wbArg_t args = wbArg_read(argc, argv);

  inputImageFile = wbArg_getInputFile(args, 1);

  inputImage = wbImport(inputImageFile);

  // The input image is in grayscale, so the number of channels
  // is 1
  imageWidth  = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);

  // Since the image is monochromatic, it only contains only one channel
  outputImage = wbImage_new(imageWidth, imageHeight, 0);

  hostInputImageData  = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  cudaMalloc((void **)&deviceInputImageData,
             imageWidth * imageHeight * sizeof(float));
  cudaMalloc((void **)&deviceOutputImageData,
             imageWidth * imageHeight * sizeof(float));
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  cudaMemcpy(deviceInputImageData, hostInputImageData,
             imageWidth * imageHeight * sizeof(float),
             cudaMemcpyHostToDevice);
  wbTime_stop(Copy, "Copying data to the GPU");

  ///////////////////////////////////////////////////////
  wbTime_start(Compute, "Doing the computation on the GPU");
  dim3 local_size(32, 32, 1);
  dim3 global_size(imageHeight/32 +1, imageWidth/32 +1, 1);

  blur<<<local_size, global_size>>> (deviceInputImageData, deviceOutputImageData, imageHeight, imageWidth);

  wbTime_stop(Compute, "Doing the computation on the GPU");

  ///////////////////////////////////////////////////////
  wbTime_start(Copy, "Copying data from the GPU");
  cudaMemcpy(hostOutputImageData, deviceOutputImageData,
             imageWidth * imageHeight * sizeof(float),
             cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  wbSolution(args, outputImage);

  cudaFree(deviceInputImageData);
  cudaFree(deviceOutputImageData);

  wbImage_delete(outputImage);
  wbImage_delete(inputImage);

  return 0;
}

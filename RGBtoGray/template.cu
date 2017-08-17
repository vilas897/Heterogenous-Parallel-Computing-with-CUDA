#include<stdio.h>
#include<stdlib.h>
#include<cuda.h>
#include "wb.h"

//@@ INSERT CODE HERE
__global__ void convert(float* input, float* output, int height, int width)
{
  unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if(idx < height*width)
    output[idx] = (0.21*input[3*idx] + 0.71*input[3*idx + 1] + 0.07*input[3*idx + 2]);
}


int main(int argc, char *argv[]) {

  int imageChannels;
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

  imageWidth  = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  // For this lab the value is always 3
  imageChannels = wbImage_getChannels(inputImage);

  // Since the image is monochromatic, it only contains one channel
  outputImage = wbImage_new(imageWidth, imageHeight, 0);

  hostInputImageData  = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  cudaMalloc((void **)&deviceInputImageData,
             imageWidth * imageHeight * imageChannels * sizeof(float));
  cudaMalloc((void **)&deviceOutputImageData,
             imageWidth * imageHeight * sizeof(float));
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  cudaMemcpy(deviceInputImageData, hostInputImageData,
             imageWidth * imageHeight * imageChannels * sizeof(float),
             cudaMemcpyHostToDevice);
  wbTime_stop(Copy, "Copying data to the GPU");

  ///////////////////////////////////////////////////////
  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ INSERT CODE HERE
  convert<<<((imageWidth*imageHeight)/1024) + 1, 1024>>> (deviceInputImageData, deviceOutputImageData, imageHeight, imageWidth);

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

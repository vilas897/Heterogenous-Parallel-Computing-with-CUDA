#include<stdio.h>
#include<cuda.h>
#include<math.h>

#define BLOCK_SIZE 1024
#define NUM 8192238

//Check Error
#define printError(func)                                                \
{                                                                       \
  cudaError_t E  = func;                                                \
  if(E != cudaSuccess)                                                  \
  {                                                                     \
    printf( "\nError at line: %d ", __LINE__);                          \
    printf( "\nError:  %s ", cudaGetErrorString(E));                    \
  }                                                                     \
}                                                                       \

__global__ void reduce(float *inData, float* outData)
{
  __shared__ int sharedData[1024];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockDim.x * blockIdx.x + tid;

//  if(i>=NUM)
//    return;

  sharedData[tid] = inData[i];
  __syncthreads();

  for(i=1; i<blockDim.x; i*= 2)   //Building up the sum of all elements in a block
  {
    unsigned int idx = 2 * tid * i;

    if(idx < blockDim.x)
    {
        sharedData[idx] += sharedData[idx + i];
    }

    __syncthreads();
  }

  if(tid == 0)                       //Return the sum of the block
  {
    atomicAdd(outData, sharedData[0]);
  }
}

int main()
{
  float* arr;
  float* out;
  float* deviceInput;
  float* deviceOutput;

  arr = (float *) malloc(NUM * sizeof(float));

  int i;

  for(i=0; i<NUM; i++)                //Initialise Input data
  {
    arr[i] = 1;
  }

  int n_out;

  n_out = ceil(NUM/1024.00);
  out = (float*) malloc(sizeof(float));
  *out = 0;

  printError(cudaMalloc((void **)&deviceInput,  NUM * sizeof(float)));
  printError(cudaMalloc((void **)&deviceOutput, sizeof(float)));

//  deviceOutput[0] = 0;
  cudaMemcpy(deviceInput, arr, NUM * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceOutput, out, sizeof(float), cudaMemcpyHostToDevice);

  reduce<<<n_out, 1024>>>(deviceInput, deviceOutput);

  cudaMemcpy(out, deviceOutput, sizeof(float), cudaMemcpyDeviceToHost);

/*  for (i = 1; i < n_out; i++)
  {
      out[0] += out[i];
  }
*/
  printf("Sum = %f\n", *out);

  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  free(arr);
  free(out);
  return 0;
}

#include<stdio.h>
#include<cuda.h>

#define NUM 327133

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

//Kernel
__global__ void add(float* A, float* B, float* C)
{
  unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i<NUM)
  C[i] = B[i] + A[i];
}

//To check the output to see if it matches
int checkSum(float* A, float* B, float* C)
{
  for(int i = 0; i<NUM; i++)
    if(C[i] != A[i] + B[i])
      return 0;

  return 1;
}

int main()
{
  float* A;
  float* B;
  float* C;

  float* deviceA;
  float* deviceB;
  float* deviceC;

  A = (float*) malloc( NUM * sizeof(float));
  B = (float*) malloc( NUM * sizeof(float));
  C = (float*) malloc( NUM * sizeof(float));

  for(int i=0; i<NUM; i++)
  {
    A[i] = rand();
    B[i] = rand();
  }

  printError(cudaMalloc((void **)&deviceA,  NUM * sizeof(float)));
  printError(cudaMalloc((void **)&deviceB,  NUM * sizeof(float)));
  printError(cudaMalloc((void **)&deviceC,  NUM * sizeof(float)));

  //cudaMalloc((void **)&deviceA,  NUM * sizeof(int));
  //cudaMalloc((void **)&deviceB,  NUM * sizeof(int));
  //cudaMalloc((void **)&deviceC,  NUM * sizeof(int));

  cudaMemcpy(deviceA, A, NUM * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, B, NUM * sizeof(float), cudaMemcpyHostToDevice);

  add<<<ceil(NUM/1024.0), 1024>>>(deviceA, deviceB, deviceC);

  cudaMemcpy(C, deviceC, NUM * sizeof(float), cudaMemcpyDeviceToHost);

  if(checkSum(A, B, C))
    printf("\nResult of 2 array sum is correct\n");

   else
     printf("\nResult of 2 array sum is wrong\n");

  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);

  free(A);
  free(B);
  free(C);
}

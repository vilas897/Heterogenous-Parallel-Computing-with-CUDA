#include<stdio.h>
#include<cuda.h>

#define ROW 100
#define COL 1000
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
__global__ void add(int A[][COL], int B[][COL], int C[][COL])
{
  unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
  if( x < ROW && y < COL )
    C[x][y] = B[x][y] + A[x][y];
}

//To check the output to see if it matches
int checkSum(int A[][COL], int B[][COL], int C[][COL])
{
  for(int i = 0; i<ROW; i++)
    for(int j = 0; j<COL; j++)
      if(C[i][j] != A[i][j] + B[i][j])
        return 0;

  return 1;
}

int main()
{
  int A[ROW][COL];
  int B[ROW][COL];
  int C[ROW][COL];

  int (*deviceA)[COL];
  int (*deviceB)[COL];
  int (*deviceC)[COL];

  for(int i=0; i<ROW; i++)
  {
    for(int j=0; j<COL; j++)
    {
      A[i][j] = rand()%1000;
      B[i][j] = rand()%1000;
    }
  }

  printError(cudaMalloc((void **)&deviceA,  ROW * COL * sizeof(int)));
  printError(cudaMalloc((void **)&deviceB,  ROW * COL * sizeof(int)));
  printError(cudaMalloc((void **)&deviceC,  ROW * COL * sizeof(int)));

  cudaMemcpy(deviceA, A, ROW * COL * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, B, ROW * COL * sizeof(int), cudaMemcpyHostToDevice);

  dim3 local_size(8, 8);
  dim3 global_size(ceil(ROW/8.0), ceil(COL/8.0));

  add<<<global_size, local_size>>>(deviceA, deviceB, deviceC);

  cudaMemcpy(C, deviceC, ROW * COL * sizeof(int), cudaMemcpyDeviceToHost);

/*  for(int i=0; i<ROW; i++)
  {
    for(int j=0; j<COL; j++)
    {
      printf("%d : %d, ", A[i][j] + B[i][j], C[i][j]);
    }
    printf("\n");
  }
*/
  if(checkSum(A, B, C))
    printf("\nResult of 2 matrix sum is correct\n");

   else
     printf("\nResult of 2 matrix sum is wrong\n");

  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);
}

#include<stdio.h>
#include<math.h>
#include<cuda.h>
    
#define TILE_WIDTH 32

#define printError(func)                                                \
{                                                                       \
  cudaError_t E  = func;                                                \
  if(E != cudaSuccess)                                                  \
  {                                                                     \
    printf( "\nError at line: %d ", __LINE__);                          \
    printf( "\nError:  %s ", cudaGetErrorString(E));                    \
  }                                                                     \
}                                                                       \ 

__global__ void TiledMatrixMult(int m, int n, int k, int *A, int *B, int *C)
{
    int CValue = 0;

    int Row = blockIdx.y*TILE_WIDTH + threadIdx.y;
    int Col = blockIdx.x*TILE_WIDTH + threadIdx.x;

    __shared__ int As[TILE_WIDTH][TILE_WIDTH];
    __shared__ int Bs[TILE_WIDTH][TILE_WIDTH];

    for (int i = 0; i < (TILE_WIDTH + n - 1)/TILE_WIDTH; i++) {

         if (i*TILE_WIDTH + threadIdx.x < n && Row < m)
             As[threadIdx.y][threadIdx.x] = A[Row*n + i*TILE_WIDTH + threadIdx.x];
         else
             As[threadIdx.y][threadIdx.x] = 0.0;

         if (i*TILE_WIDTH + threadIdx.y < n && Col < k)
             Bs[threadIdx.y][threadIdx.x] = B[(i*TILE_WIDTH + threadIdx.y)*k + Col];
         else
             Bs[threadIdx.y][threadIdx.x] = 0.0;

         __syncthreads();

         for (int j = 0; j < TILE_WIDTH; ++j)
             CValue += As[threadIdx.y][j] * Bs[j][threadIdx.x];

         __syncthreads();
    }

    if (Row < m && Col < k)
        C[((blockIdx.y * blockDim.y + threadIdx.y)*k) +
           (blockIdx.x * blockDim.x)+ threadIdx.x] = CValue;
}

int checkProd(int m, int n, int k, int *A, int *B, int*C)
{
    for(int row= 0;row<m;row++)
    {
        for(int col=0;col<k;col++)
        {
            int sum=0;
            for(int i=0;i<n;i++)
            {
                sum = sum + A[row*n + i] * B[col + i*k];
            }


            if(C[row*k + col] != sum)
                return 0;
        }
    }
    return 1;
}


int main()
{
    int *A;
    int *B;
    int *C;

    int *deviceA;
    int *deviceB;
    int *deviceC;

    // Matrix A of size (m,n) and Matrix B of size (n,k)
    int m = 1024;
    int n = 512;
    int k = 1024;

    A = (int *)malloc(m * n * sizeof(int));
    B = (int *)malloc(n * k * sizeof(int));
    C = (int *)malloc(m * k * sizeof(int));

    for(int i=0;i<m*n;i++)
    {
        A[i] = rand()%10;
        //printf("%d ",A[i]);
    }
    printf("\n");

    for(int i=0;i<n*k;i++)
    {
        B[i] = rand()%10;
        //printf("%d ",B[i]);
    }
    //printf("\n");

    printError(cudaMalloc((void **)&deviceA,  m * n * sizeof(int)));
    printError(cudaMalloc((void **)&deviceB,  n * k * sizeof(int)));
    printError(cudaMalloc((void **)&deviceC,  m * k * sizeof(int)));

    cudaMemcpy(deviceA, A, m * n *  sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, B, n * k *  sizeof(int), cudaMemcpyHostToDevice);

    dim3 dimGrid((k-1)/TILE_WIDTH+1, (m-1)/TILE_WIDTH+1,1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

    //dim3 dimGrid(32,32,1);
    //dim3 dimBlock(32,32,1);

    TiledMatrixMult<<<dimGrid, dimBlock>>>(m,n,k,deviceA,deviceB,deviceC);

    cudaMemcpy(C, deviceC, m * k * sizeof(float), cudaMemcpyDeviceToHost);

    if(checkProd(m, n, k, A, B, C))
      printf("\nResult of matrix multiplication is correct\n");

    else
       printf("\nResult of matrix multiplication is wrong\n");

    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);

    free(A);
    free(B);
    free(C);
}

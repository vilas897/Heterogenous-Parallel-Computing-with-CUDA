#include<stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

int main()
{
    int i;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device name: %s\n", prop.name);
    printf("Number of multiprocessors on GPU: %d\n", prop.multiProcessorCount);
    printf("Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
    printf("Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
    printf("Peak Memory Bandwidth (GB/s): %f\n", 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    printf("Max number of blocks: %ld(%d, %d, %d)\n",(long)prop.maxGridSize[0]*prop.maxGridSize[1]*prop.maxGridSize[2],prop.maxGridSize[0],prop.maxGridSize[1],prop.maxGridSize[2]);	
    printf("Amount of shared memory per block (bytes): %ld\n", prop.sharedMemPerBlock);
    printf("Amount of global memory (bytes): %ld\n", prop.totalGlobalMem);
    printf("Warp size of GPU (number of threads): %ld\n", prop.warpSize);
    printf("Amount of constant memory (bytes): %ld\n", prop.totalConstMem);
}



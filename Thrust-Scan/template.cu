#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include "wb.h"

float* readFile(char* fileName,int *len){
    FILE *fp = fopen(fileName,"r");
    fscanf(fp,"%d",len);

    float* inp = (float*)malloc(sizeof(float)*(*len));


    for(int i=0;i<(*len);i++) fscanf(fp,"%f",&inp[i]);
    fclose(fp);

    return inp;
}
bool isEqual(float *a,float *b,int n){
    for(int i=0;i<n;i++){
        if(a[i]!=b[i]) return false;
    }
    return true;
}
int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput, *hostOutput, *deviceOutput, *expectedOutput;
  int num_elements;

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = readFile(wbArg_getInputFile(args, 0), &num_elements);
  expectedOutput = readFile(wbArg_getInputFile(args,1), &num_elements);
  hostOutput = (float*)malloc(sizeof(float)*num_elements);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ", num_elements);


  wbTime_start(GPU, "Allocating GPU memory.");
  cudaMalloc((void**)&deviceOutput,sizeof(float)*num_elements);
  cudaMemcpy(deviceOutput,hostInput,sizeof(float)*num_elements,cudaMemcpyHostToDevice);
  thrust::device_ptr<float> dev_ptr(deviceOutput);
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(Compute, "Doing the computation on the GPU");
  thrust::inclusive_scan(dev_ptr,dev_ptr+num_elements,dev_ptr);
  wbTime_stop(Compute, "Doing the computation on the GPU");

  cudaMemcpy(hostOutput,deviceOutput,sizeof(float)*num_elements,cudaMemcpyDeviceToHost);

  if(isEqual(hostOutput,expectedOutput,num_elements)) printf("Solution Verified\n");
  else printf("Wrong Solution\n");

  // Free Memory
  free(hostInput);
  free(hostOutput);
  free(expectedOutput);
  cudaFree(deviceOutput);
  return 0;
}

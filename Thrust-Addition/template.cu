#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <stdio.h>
#include <stdlib.h>
#include "wb.h"

using namespace std;

float roundthous(float num)
{
    return roundf(num*1000) /1000;
}

int main(int argc, char *argv[]) {

  wbArg_t args;
  float *hostInput1 = NULL;
  float *hostInput2 = NULL;
  float *hostOutput = NULL;
  int inputLength;

  /* parse the input arguments */
  //@@ Insert code here
  args = wbArg_read(argc, argv);
  FILE *input1 = fopen(argv[2],"r");
  FILE *input2 = fopen(argv[3],"r");
  FILE *output = fopen(argv[1],"r"); 

  fscanf(output, "%d", &inputLength);

  // Import host input data
  //@@ Read data from the raw files here
  //@@ Insert code here
  hostInput1 = (float*) malloc(inputLength*sizeof(float));
  hostInput2 = (float*) malloc(inputLength*sizeof(float));
  
  for (int i=0;i<inputLength;i++) 
  {
      fscanf(input1, "%f", &hostInput1[i]);
      fscanf(input2, "%f", &hostInput2[i]);
  }  

  // Declare and allocate host output
  //@@ Insert code here
  hostOutput = (float *) malloc(inputLength * sizeof(float));

  // Declare and allocate thrust device input and output vectors
  //@@ Insert code here 

  // Copy to device
  //@@ Insert code here
  thrust::device_vector<float> deviceInput1(hostInput1,hostInput1+inputLength);
  thrust::device_vector<float> deviceInput2(hostInput2,hostInput2+inputLength);
  thrust::device_vector<float> deviceOutput(hostOutput,hostOutput+inputLength);

  // Execute vector addition
  //@@ Insert Code here
  thrust::transform(deviceInput1.begin(), deviceInput1.end(), deviceInput2.begin(), deviceOutput.begin(), thrust::plus<float>());

  /////////////////////////////////////////////////////////

  // Copy data back to host
  //@@ Insert code here
  thrust::copy(deviceOutput.begin(), deviceOutput.end(), hostOutput);

  // Verifying the computed results
  int test = 1;
  for(int i=0;i<inputLength-1;i++)
  {
    float t;
    fscanf(output, "%f", &t);
    if(roundthous(t)!=roundthous(hostOutput[i+1]))
    {
        cout<<"Computed vector sum is wrong ";
        test=0;
        break;
    }
  } 

  if(test)
    cout<<"Computed vector sum is correct ";
  cout<<endl;
  
  free(hostInput1);
  free(hostInput2);
  free(hostOutput);
  return 0;
}

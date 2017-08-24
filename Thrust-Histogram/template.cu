#include "wb.h"

#include <thrust/adjacent_difference.h>
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sort.h>

#include<iostream>
using namespace std;

int main(int argc, char *argv[]) 
{
  wbArg_t args;
  int inputLength, num_bins;
  unsigned int *hostInput, *hostBins;

  args = wbArg_read(argc, argv);
  //FILE *output = fopen(argv[2],"r");
  //FILE *input = fopen(argv[1],"r");

  //fscanf(input, "%d", &inputLength);
  //cout<<"Input length: "<<inputLength<<endl;

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = wbImport(wbArg_getInputFile(args, 0),&inputLength);

  //hostInput = (unsigned int*) malloc(inputLength*sizeof(unsigned int));
  //for (int i=1;i<=inputLength;i++) 
  //{
  //    fscanf(input, "%d", &hostInput[i]);
  //      if(i==1 || i== inputLength)
  //          cout<<hostInput[i]<<endl;
  //}  
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The input length is ", inputLength);

  // Copy the input to the GPU
  wbTime_start(GPU, "Allocating GPU memory");
  //@@ Insert code here
  thrust::device_vector<unsigned int> deviceInput(hostInput,hostInput+inputLength);  
  wbTime_stop(GPU, "Allocating GPU memory");

  // Determine the number of bins (num_bins) and create space on the host
  //@@ insert code here
  thrust::sort(deviceInput.begin(), deviceInput.end());
  //num_bins = deviceInput.back() + 1;
  

  //cout<<deviceInput[0]<<" "<<deviceInput.back()<<deviceInput.size()<<endl;
  num_bins = deviceInput.back() + 1;
  hostBins = (unsigned int *)malloc(num_bins * sizeof(unsigned int));

  // Allocate a device vector for the appropriate number of bins
  //@@ insert code here
  thrust::device_vector<unsigned int> histogram(num_bins);

  // Create a cumulative histogram. Use thrust::counting_iterator and
  // thrust::upper_bound
  //@@ Insert code here
  thrust::counting_iterator<int> search_begin(0);
  thrust::upper_bound(deviceInput.begin(), deviceInput.end(),search_begin, search_begin + num_bins,histogram.begin());

  // Use thrust::adjacent_difference to turn the culumative histogram
  // into a histogram.
  //@@ insert code here.
  thrust::adjacent_difference(histogram.begin(), histogram.end(),histogram.begin());

  // Copy the histogram to the host
  //@@ insert code here
  thrust::copy(histogram.begin(),histogram.end(),hostBins);

  // Check the solution is correct
  wbSolution(args, hostBins, num_bins);

  // Free space on the host
  //@@ insert code here
  free(hostBins);
  free(hostInput);

  return 0;
}

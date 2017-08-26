#include "wb.h"
#include<bits/stdc++.h>
using namespace std;

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

#define mask_width 5
#define mask_radius mask_width / 2
#define TILE_WIDTH 12
#define w (TILE_WIDTH + mask_width - 1)
#define clamp(x) (min(max((x), 0.0), 1.0))



unsigned char * getArrayFromPpm(const char * filename,int *imageHeight,int *imageWidth){
  cout<<"Reading file : "<<filename<<"\n";
  string s;
  ifstream inFile(filename);
  getline(inFile,s);
  // cout<<s<<"\n";
  getline(inFile,s);
  // cout<<s<<"\n";
  inFile >> *imageWidth;
  // cout<<*imageWidth<<"\n";
  inFile >> *imageHeight;
  // cout<<*imageHeight<<"\n";
  int t = 0;
  inFile >> t;
  // cout<<t<<"\n";
  int temp;
  unsigned char * conts = (unsigned char *) malloc((*imageHeight) * (*imageWidth) * 3 * sizeof(unsigned char));
  int i=0;
  while(inFile >> temp){
    conts[i++] = temp;
  }
  cout<<"Read finished\n";
  inFile.close();
  return conts; 
}
float * getArrayFromMask(const char * filename){
  cout<<"Reading file : "<<filename<<"\n";
  ifstream inFile(filename);
  float temp;
  float * conts = (float *) malloc(5 * 5 * sizeof(float));
  int i=0;
  while(inFile>>temp){
  	// inFile.read((char*)&temp, sizeof(float));
    // cout<<temp<<" temp\n";
    conts[i++] = temp;
  }

  cout<<"Read finished\n";
  inFile.close();
  return conts; 
}

void wbSolution(const wbArg_t& args, const unsigned char* image)
{
	int imageWidth,imageHeight;
    unsigned char * solnImage = getArrayFromPpm(wbArg_getInputFile(args,0),&imageHeight,&imageWidth);
        cout<<imageHeight<<" "<<imageWidth<<"\n";
        int errCnt = 0;

        for (int i = 0; i < imageHeight*imageWidth*3; ++i)
        {
                    // cout<<solnImage[i]<<" : "<<image[i]<<"\n";
        			// const float error = fabs(solnImage[i] - image[i]);
                    if (solnImage[i] != image[i])
                    {
                        if (errCnt < wbInternal::kErrorReportLimit)
                            std::cout << "Image pixels do not match at position (" << i << "). [" << (int)image[i] << ", " <<  (int)solnImage[i] << "]\n";
                        ++errCnt;
                    }
        }

        if (!errCnt)
            std::cout << "Solution is correct." << std::endl;
        else
            std::cout << errCnt << " tests failed!" << std::endl;
    // }

    // wbImage_delete(solnImage);
}

//@@ INSERT CODE HERE


// unsigned char * compute(unsigned char *data, float *mask, int height,
//              int width) {

//   const int num_channels = 3;

//   float inputData[height * width * num_channels];
//   for(int i =0 ;i<height*width*num_channels;++i){
//       inputData[i] = ((int)data[i])/255.0;
//   }

//   float *outputData = (float *) malloc(height*width*3*sizeof(float));

//   int img_width  = width;
//   int img_height = height;
//   int mask_rows = 5;
//   int mask_cols = 5;
//   int mask_radius_y = mask_rows / 2; // 5 X 5 mask matrix is fixed
//   int mask_radius_x = mask_cols / 2;
//   for (int out_y = 0; out_y < img_height; ++out_y) {
//     for (int out_x = 0; out_x < img_width; ++out_x) {
//       for (int c = 0; c < num_channels; ++c) { // channels
//         float acc = 0;
//         for (int off_y = -mask_radius_y; off_y <= mask_radius_y; ++off_y) {
//           for (int off_x = -mask_radius_x; off_x <= mask_radius_x;
//                ++off_x) {
//             int in_y   = out_y + off_y;
//             int in_x   = out_x + off_x;
//             int mask_y = mask_radius_y + off_y;
//             int mask_x = mask_radius_x + off_x;
//             if (in_y < img_height && in_y >= 0 && in_x < img_width &&
//                 in_x >= 0) {
//               acc +=
//                   (inputData[(in_y * img_width + in_x) * num_channels + c]) *
//                   mask[mask_y * mask_cols + mask_x];
//                   // cout<<mask[mask_y * mask_cols + mask_x]<<"\n";
//             } else {
//               acc += 0.0f;
//             }
//           }
//         }
//         // fprintf(stderr, "%f %f\n", clamp(acc));
//         cout<<"acc : "<<acc<<"\n";
//         outputData[(out_y * img_width + out_x) * num_channels + c] =
//             clamp(acc);
//       }
//     }
//   }
//   unsigned char *output = (unsigned char *) malloc(height*width*3*sizeof(unsigned char));
//   for(int i =0;i<height*width*num_channels;++i){
//       output[i] = (unsigned char) floor(outputData[i] * 255);
//       // cout<<i<<" helloo "<<outputData[i]<<"\n";
//   }
//   return output;
// }

static void write_data(const char *file_name, unsigned char *data,
                       unsigned int width, unsigned int height,
                       unsigned int channels) {
    FILE *handle = fopen(file_name, "w");
    fprintf(handle, "P6\n");
    fprintf(handle, "#Created by %s\n", __FILE__);
    fprintf(handle, "%d %d\n", width, height);
    fprintf(handle, "255\n");
    for(int i=0;i<width*height*channels;++i){
    	fprintf(handle,"%d ",data[i]);
    }
  fflush(handle);
  fclose(handle);
}


// // using global memory kernel

// __global__ 
// void convolution(float * deviceInputImageData, float *deviceMaskData,unsigned char *deviceOutputImageData,int imageChannels,int imageWidth,int imageHeight){
// 	int y = blockIdx.y*blockDim.y + threadIdx.y;
// 	int x = blockIdx.x*blockDim.x + threadIdx.x;

//     if(y < imageHeight && x < imageWidth){

//         	float acc = 0;
// 	        for (int off_y = -mask_radius; off_y <= mask_radius; ++off_y) {
// 	          for (int off_x = -mask_radius; off_x <= mask_radius;++off_x) {
// 	            int in_y   = y + off_y;
// 	            int in_x   = x + off_x;
// 	            int mask_y = mask_radius + off_y;
// 	            int mask_x = mask_radius + off_x;
// 	            if (in_y < imageHeight && in_y >= 0 && in_x < imageWidth && in_x >= 0) {
// 	              acc += deviceInputImageData[(in_y * imageWidth + in_x)] * deviceMaskData[mask_y * mask_width + mask_x];
// 	            } else {
// 	              acc += 0.0f;
// 	            }
// 	          }
// 	        }
// 	        // deviceOutputImageData[(y * imageWidth + x) * imageChannels + c] = (deviceInputImageData[(y * imageWidth + x) * imageChannels + c]);    

// 	        deviceOutputImageData[(y * imageWidth + x)] = (unsigned char)(floor(clamp(acc)*255));    
//     }
// }

__global__ 
void convolution(float * deviceInputImageData, const float * __restrict__ deviceMaskData,unsigned char *deviceOutputImageData,int imageChannels,int imageWidth,int imageHeight){
	int y = blockIdx.y * TILE_WIDTH + threadIdx.y - mask_radius;
	int x = blockIdx.x * TILE_WIDTH + threadIdx.x - mask_radius;																	

	__shared__ float ds_Image[16][16];
	
	ds_Image[threadIdx.y][threadIdx.x] = 0;

	if(y>=0 && y< imageHeight && x>=0 && x < imageWidth)
	{
		ds_Image[threadIdx.y][threadIdx.x] = deviceInputImageData[y* imageWidth + x];

		__syncthreads();

       	float acc = 0;

       	if(threadIdx.x >= mask_radius && threadIdx.x < 16 - mask_radius && threadIdx.y >= mask_radius
       	 	&& threadIdx.y < 16 - mask_radius){

			    for (int off_y = -mask_radius; off_y <= mask_radius; ++off_y) 
			    {
			        for (int off_x = -mask_radius; off_x <= mask_radius;++off_x) 
			        {
			            int in_y   = threadIdx.y + off_y;
			            int in_x   = threadIdx.x + off_x;
			            int mask_y = mask_radius + off_y;
			            int mask_x = mask_radius + off_x;
			            if (in_y < 16 && in_y >= 0 && in_x < 16 && in_x >= 0) 
			            {
			              acc += ds_Image[in_y][in_x] * deviceMaskData[mask_y * mask_width + mask_x];
			            } 
		            	 else 
		            	{
		              		acc += 0.0f;
		            	}
			          }
			    }

			    // __syncthreads();
			        // deviceOutputImageData[(y * imageWidth + x) * imageChannels + c] = (deviceInputImageData[(y * imageWidth + x) * imageChannels + c]);    

			    deviceOutputImageData[(y * imageWidth + x)] = (unsigned char)(floor(clamp(acc)*255.0));    
	    }
    }
}

__global__
void split(float *ip, float *red_channel, float *green_channel,float *blue_channel, int width, int height)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int row = by*blockDim.y + ty;
	int col = bx*blockDim.x + tx;


	if(row<height && col<width)
	{
		red_channel[row*width + col] = ip[3*(row*width+col)];
		green_channel[row*width + col] = ip[3*(row*width+col) + 1];
		blue_channel[row*width + col] = ip[3*(row*width+col) + 2];
	}

}

__global__
void mergeColors(unsigned char *red_channel, unsigned char *green_channel,unsigned char *blue_channel, unsigned char *output, int width, int height)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int row = by*blockDim.y + ty;
	int col = bx*blockDim.x + tx;


	if(row<height && col<width)
	{
		output[3*(row*width + col)] = red_channel[row*width + col];
		output[3*(row*width + col)+1] = green_channel[row*width + col];
		output[3*(row*width + col)+2] = blue_channel[row*width + col];
	} 

}


int main(int argc, char *argv[]) {
  wbArg_t arg;
  int maskRows;
  int maskColumns;
  int imageChannels;
  int imageWidth;
  int imageHeight;
  char *inputImageFile;
  char *inputMaskFile;
  // wbImage_t inputImage;
  // wbImage_t outputImage;
  unsigned char *tempInputImage;
  float *hostInputImageData;
  unsigned char *hostOutputImageData;
  float *hostMaskData;
  float *deviceInputImageData;
  unsigned char *deviceOutputImageData;
  float *deviceMaskData;

  float *deviceInputRed;
  float *deviceInputGreen;
  float *deviceInputBlue;

  unsigned char *deviceOutputRed;
  unsigned char *deviceOutputGreen;
  unsigned char *deviceOutputBlue;


  arg = wbArg_read(argc, argv); /* parse the input arguments */


  inputImageFile = wbArg_getInputFile(arg, 1);
  inputMaskFile  = wbArg_getInputFile(arg, 2);

  hostMaskData = getArrayFromMask(inputMaskFile);
  // cout<<"MAsk : \n";
  // for(int i =0;i<5;++i){
  // 	for(int j=0;j<5;++j){
  // 		cout<<hostMaskData[i*5+j]<<" ";
  // 	}
  // 	cout<<"\n";
  // }
  tempInputImage =  getArrayFromPpm(inputImageFile,&imageHeight,&imageWidth);
  // cout<<(int)tempInputImage[0]<<"\n";
  imageChannels = 3;
  

  int imageSize = imageHeight*imageWidth*imageChannels;
  hostInputImageData = (float*) malloc(imageSize*sizeof(float));

  for(int i=0;i<imageWidth*imageHeight*imageChannels;++i){
  	hostInputImageData[i] = ((int)tempInputImage[i])/255.0;
  }

  hostOutputImageData = (unsigned char *) malloc(imageSize*sizeof(unsigned char));
  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");


  wbTime_start(GPU, "Doing GPU memory allocation");
  wbCheck(cudaMalloc((void **)&deviceInputImageData,imageSize*sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceMaskData,5*5*sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutputImageData,imageSize*sizeof(unsigned char)));

  wbCheck(cudaMalloc((void **)&deviceInputRed,imageHeight*imageWidth*sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceInputGreen,imageHeight*imageWidth*sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceInputBlue,imageHeight*imageWidth*sizeof(float)));

  wbCheck(cudaMalloc((void **)&deviceOutputRed,imageHeight*imageWidth*sizeof(unsigned char)));
  wbCheck(cudaMalloc((void **)&deviceOutputGreen,imageHeight*imageWidth*sizeof(unsigned char)));
  wbCheck(cudaMalloc((void **)&deviceOutputBlue,imageHeight*imageWidth*sizeof(unsigned char)));



  cudaMemset(deviceOutputImageData,0,imageSize*sizeof(unsigned char));
  // //@@ INSERT CODE HERE
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  // //@@ INSERT CODE HERE
  wbCheck(cudaMemcpy(deviceInputImageData,hostInputImageData,imageSize*sizeof(float),cudaMemcpyHostToDevice));
  // wbCheck(cudaMemcpy(deviceOutputImageData,hostOutputImageData,imageSize*sizeof(float),cudaMemcpyHostToDevice));
  wbCheck(cudaMemcpy(deviceMaskData,hostMaskData,5*5*sizeof(float),cudaMemcpyHostToDevice));
  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  dim3 dimBlock(16,16,1);
  int gridsx = (imageWidth-1)/TILE_WIDTH + 1;
  int gridsy = (imageHeight-1)/TILE_WIDTH + 1;

  // dim3 dimGrid(2,2,1);
  dim3 dimGrid(gridsx,gridsy,1);
  
  // cout<<hostInputImageData<<" : "<<

  split<<<dimGrid,dimBlock>>>(deviceInputImageData,deviceInputRed,deviceInputGreen,
  								deviceInputBlue,imageWidth,imageHeight);

  convolution<<<dimGrid,dimBlock>>>(deviceInputRed, deviceMaskData,
                                     deviceOutputRed, imageChannels,
                                     imageWidth, imageHeight);
  convolution<<<dimGrid,dimBlock>>>(deviceInputGreen, deviceMaskData,
                                     deviceOutputGreen, imageChannels,
                                     imageWidth, imageHeight);
  convolution<<<dimGrid,dimBlock>>>(deviceInputBlue, deviceMaskData,
                                     deviceOutputBlue, imageChannels,
                                     imageWidth, imageHeight);

  mergeColors<<<dimGrid,dimBlock>>>(deviceOutputRed,deviceOutputGreen,deviceOutputBlue,
  									 deviceOutputImageData,imageWidth,imageHeight);

  // hostOutputImageData =  compute(tempInputImage,hostMaskData,imageHeight,imageWidth);
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  // //@@ INSERT CODE HERE
  wbCheck(cudaMemcpy(hostOutputImageData, deviceOutputImageData,
             imageWidth * imageHeight * imageChannels * sizeof(unsigned char),
             cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  // cout<<hostInputImageData[0]<<" : "<<(int)hostOutputImageData[0]<<"\n";

  wbSolution(arg, hostOutputImageData); //changed def in wb.h

  // //@@ Insert code here

  // free(hostMaskData);
  // wbImage_delete(outputImage);
  // wbImage_delete(inputImage);

  return 0;
}

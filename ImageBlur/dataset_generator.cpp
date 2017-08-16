#include<stdio.h>
#include<stdlib.h>
#include<string.h>

#define CHANNELS 3
#define BLUR_SIZE 1

static char *base_dir;

static void compute(unsigned char *out, unsigned char *in,
                    unsigned int height, unsigned int width) {
  for (unsigned int row = 0; row < height; row++) {
    for (unsigned int col = 0; col < width; col++) {
      for(unsigned int channels = 0; channels < 3; channels++)
      {
        int pixVal = 0;
        int pixels = 0;
        // Get the average of the surrounding BLUR_SIZE x BLUR_SIZE box
        for (int blurrow = -BLUR_SIZE; blurrow < BLUR_SIZE + 1; ++blurrow) {
          for (int blurcol = -BLUR_SIZE; blurcol < BLUR_SIZE + 1;
              ++blurcol) {
                int currow = row + blurrow;
                int curcol = col + blurcol;
                // Verify we have a valid image pixel
                if (currow > -1 && static_cast<unsigned int>(currow) < height &&
                curcol > -1 && static_cast<unsigned int>(curcol) < width) {
                  pixVal += in[3*(currow * width + curcol) + channels];
                  pixels++; // Keep track of number of pixels in the avg
                }
              }
      }


    out[3*(row * width + col) + channels] = (unsigned char)(pixVal / pixels);
      // Write our new pixel value out
    }
    }
  }
}

static unsigned char *generate_data(const unsigned int y,
                                    const unsigned int x) {
  /* raster of y rows
     R, then G, then B pixel
     if maxVal < 256, each channel is 1 byte
     else, each channel is 2 bytes
  */
  unsigned int i;

  const int maxVal    = 255;
  unsigned char *data = (unsigned char *)malloc(y * x * 3);

  unsigned char *p = data;
  for (i = 0; i < y * x; ++i) {
    unsigned short r = rand() % maxVal;
    unsigned short g = rand() % maxVal;
    unsigned short b = rand() % maxVal;
    *p++             = r;
    *p++             = g;
    *p++             = b;
  }
  return data;
}

static void write_data(char *file_name, unsigned char *data,
                       unsigned int width, unsigned int height,
                       unsigned int channels) {
  FILE *handle = fopen(file_name, "w");
  if (channels == 1) {
    fprintf(handle, "P5\n");
  } else {
    fprintf(handle, "P6\n");
  }
  fprintf(handle, "#Created by %s\n", __FILE__);
  fprintf(handle, "%d %d\n", width, height);
  fprintf(handle, "255\n");

  fwrite(data, width * channels * sizeof(unsigned char), height, handle);

  fflush(handle);
  fclose(handle);
}

static void create_dataset(const int datasetNum, const int y,
                           const int x) {

  //char* integer_string;
  //sprintf(integer_string, "%d", datasetNum);
  char* input_file_name  = "input.ppm";
  char* output_file_name = "output.ppm";

  //strcat(input_file_name, integer_string);
  //strcat(output_file_name, integer_string);

  //char *ext = ".ppm";

  //strcat(input_file_name, ext);
  //strcat(output_file_name, ext);

  unsigned char *input_data = generate_data(y, x);
  unsigned char *output_data =
      (unsigned char *)calloc(sizeof(unsigned char), y * x * CHANNELS);

  compute(output_data, input_data, y, x);

  write_data(input_file_name, input_data, x, y, CHANNELS);
  write_data(output_file_name, output_data, x, y, CHANNELS);

  free(input_data);
  free(output_data);
}

int main() {

  create_dataset(1, 512, 512);

  return 0;
}

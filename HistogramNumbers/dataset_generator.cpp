#include <bits/stdc++.h>

using namespace std;

static char *base_dir;
const size_t NUM_BINS      = 4096;
const unsigned int BIN_CAP = 127;

static void compute(unsigned int *bins, unsigned int *input, int num) {
  for (int i = 0; i < num; ++i) {
    int idx = input[i];
    if (bins[idx] < BIN_CAP)
      ++bins[idx];
  }
}

static unsigned int *generate_data(size_t n, unsigned int num_bins) {
  unsigned int *data = (unsigned int *)malloc(sizeof(unsigned int) * n);
  for (unsigned int i = 0; i < n; i++) {
    data[i] = rand() % num_bins;
  }
  return data;
}

static void write_data(const char *file_name, unsigned int *data, int num) {
  FILE *handle = fopen(file_name, "w");
  fprintf(handle, "%d", num);
  for (int ii = 0; ii < num; ii++) {
    fprintf(handle, "\n%d", *data++);
  }
  fflush(handle);
  fclose(handle);
}

static void create_dataset(int datasetNum, size_t input_length, size_t num_bins) {

  std::string x;
  x.push_back(datasetNum + '0');

  //const char *dir_name =
     // wbDirectory_create(wbPath_join(base_dir, x.c_str()));
                     //
  string y = x;
  x.append("input.raw");
  y.append("output.raw");

  //char *input_file_name  = x.c_str();				
  //char *output_file_name = y.c_str();					

  unsigned int *input_data = generate_data(input_length, num_bins);
  unsigned int *output_data =
      (unsigned int *)calloc(sizeof(unsigned int), num_bins);

  compute(output_data, input_data, input_length);


  write_data(x.c_str(), input_data, input_length);

  write_data(y.c_str(), output_data, num_bins);

  free(input_data);
  free(output_data);
  
}

int main() {
  //base_dir = wbPath_join(wbDirectory_current(), "Histogram", "Dataset");

  create_dataset(0, 16, NUM_BINS);
  create_dataset(1, 1024, NUM_BINS);
  create_dataset(2, 513, NUM_BINS);
  create_dataset(3, 511, NUM_BINS);
  create_dataset(4, 1, NUM_BINS);
  create_dataset(5, 500000, NUM_BINS);
  return 0;
}

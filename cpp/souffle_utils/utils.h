#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <cstdlib>
#include <time.h> 
#include <sstream>      // std::stringstream

#include <cuda_fp16.h>

// typedef unsigned long uint64_t;

unsigned long get_shape_size(std::vector<unsigned long> shape){
    unsigned long shape_size = 1;
    for(auto s: shape){
        shape_size *= s;
    }
    return shape_size;
}

void print_and_check(std::vector<float>& output, float expected){
  for(int i=0; i<16; ++i){
    printf("%f ", output[i]);
  }printf("\n");
  for(int i=0; i<output.size(); ++i){
    auto d = output[i];
    if(d!=expected){
      printf("error %d output is %f\n", i, d);
      break;
    }
  }
}


template<typename T>
void hf_init_values(T* input, std::vector<int> shape, T value, int mod=0){
  srand (time(NULL));
  if(shape.size()==1){
    for(int i=0; i<shape[0]; ++i){
      if(mod==0){
        input[i] = value;
        // input[i] = __float2half(((float)(i % 10)) / 10 / 32);
      }else{
        input[i] = __float2half((float)(rand() % 20) / 100);
      }
    }
  }else{
    int left_size = 1;
    std::vector<int> new_shape;
    for(int i=1; i<shape.size(); ++i){
      left_size *= shape[i];
      new_shape.push_back(shape[i]);
    }
    for(int i=0; i<shape[0]; ++i){
      init_values(input+i*left_size, new_shape, value, mod);
    }
  }
}



// mod: 0 for fix number 1 for random number
template<typename T>
void init_values(T* input, std::vector<int> shape, T value, int mod=0){
  srand (time(NULL));
  if(shape.size()==1){
    for(int i=0; i<shape[0]; ++i){
      if(mod==0){
        input[i] = value;
      }else{
        input[i] = (T)(rand() % 10);
        // if(i%2==0){
        //   input[i] = 1;
        // }else{
        //   input[i] = 2;
        // }
      }
    }
  }else{
    int left_size = 1;
    std::vector<int> new_shape;
    for(int i=1; i<shape.size(); ++i){
      left_size *= shape[i];
      new_shape.push_back(shape[i]);
    }
    for(int i=0; i<shape[0]; ++i){
      init_values(input+i*left_size, new_shape, value, mod);
    }
  }
}

void test_init_values(){
  int n=1, c=3, h=4, w =4;
  float* input = new float[n*c*h*w];
  init_values<float>(input, {n,c,h,w}, 1, 1);
  for(int i=0;i<n*c*h*w; ++i){
    printf("%.2f ", input[i]);
  }
}

/**
 * @brief Init input and weights for two continuos conv
 * 
 * @param input 
 * @param dw_weight weight for first depthwise conv
 * @param pw_weight weight for second pointwise conv
 * @param output the output
 * @param height the input img height
 * @param width the input img width
 * @param in_channel the input_channel of the img
 * @param out_channel the output_channel of the output
 * @param kernel_height the depthwise kernel_height
 * @param kernel_width the depthwise kernel width
 */
void init_conv_conv_fusion_data(float* input, float* weight1, float* weight2, float* output,
  const int height, const int width,
  const int kernel1_height, const int kernel1_width, const int kernel1_in_channel, const int kernel1_out_channel, 
  const int kernel2_height, const int kernel2_width, const int kernel2_in_channel, const int kernel2_out_channel){
  init_values<float>(input, {height, width, kernel1_in_channel}, 1);
  init_values<float>(weight1, {kernel1_height, kernel1_width, kernel1_in_channel, kernel1_out_channel}, 1);
  init_values<float>(weight2, {kernel2_height, kernel2_width, kernel2_in_channel, kernel2_out_channel}, 1);
}
// /**
//  * @brief Init input and weights for two continuos conv
//  * 
//  * @param input 
//  * @param dw_weight weight for first depthwise conv
//  * @param pw_weight weight for second pointwise conv
//  * @param output the output
//  * @param height the input img height
//  * @param width the input img width
//  * @param in_channel the input_channel of the img
//  * @param out_channel the output_channel of the output
//  * @param kernel_height the depthwise kernel_height
//  * @param kernel_width the depthwise kernel width
//  */
// void init_conv_conv_fusion_data(float* input, float* weight1, float* weight2, float* output,
//   const int height, const int width,
//   const int kernel1_height, const int kernel1_width, const int kernel1_in_channel, const int kernel1_out_channel, 
//   const int kernel2_height, const int kernel2_width, const int kernel2_in_channel, const int kernel2_out_channel){
//     srand (time(NULL));
//   for (int h = 0; h < height; ++h) {
//       for (int w = 0; w < width; ++w) {
//         for (int ic = 0; ic < kernel1_in_channel; ++ic) {
//         input[h*width*kernel1_in_channel + w*kernel1_in_channel + ic] = rand() % 10;
//         // input[h*width*kernel1_in_channel + w*kernel1_in_channel + ic] = 1;
//         // if(ic%2==0){input[h*width*in_channel + w*in_channel + ic] = 1;}
//         // else{input[h*width*in_channel + w*in_channel + ic] = 2;}
//         // input[h*width*kernel1_in_channel + w*kernel1_in_channel + ic] = rand() % 10;
//       }
//     }
//   }
  
//   for (int h = 0; h < kernel1_height; ++h) {
//     for (int w = 0; w < kernel1_width; ++w) {
//       for (int ic = 0; ic < kernel1_in_channel; ++ic) {
//         for(int oc = 0; oc < kernel1_out_channel; ++oc) {
//           weight1[h * kernel1_width * kernel1_in_channel * kernel1_out_channel 
//             + w * kernel1_in_channel * kernel1_out_channel + ic * kernel1_out_channel + oc] = rand() % 10;
//         }
//       }
//     }
//   }

//   for (int h = 0; h < kernel2_height; ++h) {
//     for (int w = 0; w < kernel2_width; ++w) {
//       for (int ic = 0; ic < kernel2_in_channel; ++ic) {
//         for(int oc = 0; oc < kernel2_out_channel; ++oc) {
//           weight2[h * kernel2_width * kernel2_in_channel * kernel2_out_channel 
//             + w * kernel2_in_channel * kernel2_out_channel + ic * kernel2_out_channel + oc] = rand() % 10;
//         }
//       }
//     }
//   }

//   for (int h = 0; h < height; ++h) {
//       for (int w = 0; w < width; ++w) {
//         for (int oc = 0; oc < kernel2_out_channel; ++oc) {
//         output[h*width*kernel2_out_channel + w*kernel2_out_channel + oc] = 0;
//       }
//     }
//   }
// }


void pointwise_conv(float* input, float* pw_weight, float* output,
  const int height, const int width, const int in_channel, const int out_channel){
  for (int h = 0; h < height; ++h) {
    for (int w = 0; w < width; ++w) {
      for(int oc = 0; oc<out_channel; ++oc){
        float sum = 0;
        for (int ic = 0; ic < in_channel; ++ic) {
          // sum += input[h*width*in_channel + w*in_channel + ic] * pw_weight[oc*in_channel + ic];
          sum += input[h*width*in_channel + w*in_channel + ic] * pw_weight[ic*out_channel+oc];
        }
        output[h*width*out_channel+ w*out_channel + oc] = sum;
      }
    }
  }
}
  
bool check_equal(float* output1, float* output2, int height, int width, int channels){
  std::stringstream ss;
  bool equal = true;
  for (int h = 0; h < height; ++h) {
    for (int w = 0; w < width; ++w) {
      for (int oc = 0; oc < channels; ++oc) {
        int idx = h*width*channels + w*channels+oc;
        if(std::abs(output1[idx] - output2[idx]) > 0.1 ){
          equal = false;
          ss << "<" << h << "," << w <<"," << oc <<",> " << output1[idx] << " " << output2[idx] <<"\n";
        }
      }
    }
  }
  if(!equal){
    printf("Check failed!\n");
    printf("%s", ss.str().c_str());  
  }else{
    printf("Check pass!\n");
  }
  return equal;
}

void print_output(float* output, int height, int width, int channels){
  printf("outputs:->\n");
  for (int h = 0; h < height; ++h) {
    for (int w = 0; w < width; ++w) {
      for (int oc = 0; oc < channels; ++oc) {
        int idx = h*width*channels + w*channels+oc;
        printf("%.2f ", output[idx]);
      }printf("\n");
    }printf("\n");
  }printf("\n");
}


#endif
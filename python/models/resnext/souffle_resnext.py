
import sys, os
import logging
FORMAT = '%(asctime)s %(filename)s:%(lineno)d %(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO)
sys.path.extend(['', '/home/xiachunwei/Software/clean_tvm/tvm/python/tvm/contrib/',
'/home/xiachunwei/Software/clean_tvm/tvm/python',
 '/home/xiachunwei/Software/tensor-compiler/src/itvm/operator_fusion/models/resnext', 
 '/home/xiachunwei/Software/anaconda3/lib/python37.zip', '/home/xiachunwei/Software/anaconda3/lib/python3.7', 
 '/home/xiachunwei/Software/anaconda3/lib/python3.7/lib-dynload', '/home/xiachunwei/.local/lib/python3.7/site-packages', 
 '/home/xiachunwei/Software/anaconda3/lib/python3.7/site-packages', 
 '/home/xiachunwei/Projects/CenterNet/src/lib/models/networks/DCNv2', '/home/xiachunwei/Projects/tensor-compiler-gpu/src/transform_preds', '/home/xiachunwei/Projects/EfficientNet-PyTorch', '/home/xiachunwei/Software/anaconda3/lib/python3.7/site-packages/bert_binding-0.0.0-py3.7-linux-x86_64.egg', '/home/xiachunwei/Software/anaconda3/lib/python3.7/site-packages/lstm_binding-0.0.0-py3.7-linux-x86_64.egg'])
sys.path.append("/home/xiachunwei/Software/clean_tvm/tvm/python/")
sys.path.append(os.path.dirname(os.path.abspath(__file__))+os.sep+"../../")
from ansor_module import AnsorModule

from resnext_kernels import fused_conv1x1_bn_relu, fused_conv1x1_bn_relu_stride, \
  fused_conv3x3, conv1x1, fused_conv7x7_s2_bn_relu, avgpool, conv3x3, conv1x1_bn_relu, maxpool


# This implementation is based on the nnfusion/artifacts/models/resnext_imagenet_nchw
# git clone https://github.com/microsoft/nnfusion.git && git checkout osdi20_artifact
# cd nnfusion/artifacts/models/resnext_imagenet_nchw
class ResNext(AnsorModule):
  def __init__(self, image_size, num_residual, fuse=True, num_bench=1, num_repeat=1) -> None:
    super().__init__(num_bench=num_bench, num_repeat=num_repeat)
    self.image_size = image_size
    self.num_residual = num_residual
    self.fuse = fuse
  
  def residual_layer(self, batch, height, width, in_channels, out_channels, stride=1, num_parts=64, num_repeats=1, print_source=False):
    layer_latency_arr = []
    residule_num_of_kernels = 0
    for _ in range(num_repeats):
      # Split layer
      # First layer, only accounts for in_channels -> out_channels, do not stride
      if stride==1:
        if self.fuse:
          splited_channels = out_channels // num_parts
          config = [batch, height, width, in_channels, splited_channels, num_parts]
          log_file = "kernel_configs/resnext_imagenet_50_fused_conv1x1_bn_relu_{}_{}_{}_{}_{}_{}.log".format(*config)
          self.apply(fused_conv1x1_bn_relu, config, log_file, print_source=print_source)
          residule_num_of_kernels+=1
        else:
          splited_channels = out_channels // num_parts
          config = [batch, height, width, in_channels, splited_channels, stride]
          log_file = "kernel_configs/resnext_imagenet_50_conv1x1_bn_relu_{}_{}_{}_{}_{}_{}.log".format(*config)
          [self.apply(conv1x1_bn_relu, config, log_file, print_source=print_source) for i in range(num_parts)]
          residule_num_of_kernels+=num_parts
      else: # stride==2
        if self.fuse:
          splited_channels = out_channels // num_parts
          config = [batch, height, width, in_channels, splited_channels, num_parts, stride]
          log_file = "kernel_configs/resnext_imagenet_50_fused_conv1x1_bn_relu_{}_{}_{}_{}_{}_{}_s{}.log".format(*config)
          self.apply(fused_conv1x1_bn_relu_stride, config, log_file)
          residule_num_of_kernels+=1
        else:
          splited_channels = out_channels // num_parts
          config = [batch, height, width, in_channels, splited_channels, stride]
          log_file = "kernel_configs/resnext_imagenet_50_conv1x1_bn_relu_{}_{}_{}_{}_{}_{}.log".format(*config)
          [self.apply(conv1x1_bn_relu, config, log_file, print_source=print_source) for i in range(num_parts)]
          residule_num_of_kernels+=num_parts
      # Second layer, input and output channel always same, may be stride
      if self.fuse:
        config = [batch, height, width, splited_channels, splited_channels, 3, 3, num_parts]
        log_file = "kernel_configs/resnext_imagenet_50_fused_conv3x3_{}_{}_{}_{}_{}_{}.log".format(*config)
        self.apply(fused_conv3x3, config, log_file, print_source=print_source)
        residule_num_of_kernels+=1
      else:
        config = [batch, height, width, splited_channels, splited_channels, 3, 3]
        log_file = "kernel_configs/resnext_imagenet_50_conv3x3_{}_{}_{}_{}_{}_{}_{}.log".format(*config)
        [self.apply(conv3x3, config, log_file, print_source=print_source) for i in range(num_parts)]
        residule_num_of_kernels+=num_parts

      # transition_layer
      # Third layer, always same as out_channels
      config = [batch, height, width, out_channels, out_channels]
      log_file = "kernel_configs/resnext_imagenet_50_conv1x1_{}_{}_{}_{}_{}.log".format(*config)
      self.apply(conv1x1, config, log_file, print_source=print_source)
      residule_num_of_kernels += 1

      # Last layer of one stage, only with stride==2, need to downsample
      # Need fuse add+relu here
      if in_channels != out_channels:
        config = [batch, height, width, in_channels, out_channels]
        log_file = "kernel_configs/resnext_imagenet_50_conv1x1_{}_{}_{}_{}_{}.log".format(*config)
        self.apply(conv1x1, config, log_file)
        residule_num_of_kernels += 1

    self.latency_array.extend(layer_latency_arr * num_repeats)
    self.num_of_kernels += (residule_num_of_kernels * num_repeats)


  def forward(self):
    # First layer: conv([1,224,224,3] * [7, 7, 3, 64], stride=2) + bn + relu + maxpool([3*3], stride=2)
    # input: [1,224,224,3], output: [1,56, 56, 64]
    config = [1, 224, 224, 3, 64, 7, 7]
    log_file = "kernel_configs/resnext_imagenet_50_fused_conv7x7_{}_{}_{}_{}_{}_{}_{}.log".format(*config)
    self.apply(fused_conv7x7_s2_bn_relu, config, log_file)
    
    config = [1, 112, 112, 64, 3, 3, 2]
    log_file = "kernel_configs/resnext_imagenet_50_maxpool_{}_{}_{}_{}_{}_{}_{}.log".format(*config)
    self.apply(maxpool, config, log_file)
    logging.info("conv1 finish!")
    # Conv2
    self.residual_layer(1, 56, 56, 64, 256)
    self.residual_layer(1, 56, 56, 256, 256, num_repeats=self.num_residual[0]-1)
    logging.info("conv2 finish!")
    self.residual_layer(1, 56, 56, 256, 512, stride=2)
    # Conv3
    self.residual_layer(1, 28, 28, 512, 512, num_repeats=self.num_residual[1]-1)
    self.residual_layer(1, 28, 28, 512, 1024, stride=2)
    logging.info("conv3 finish!")
    # Conv4
    self.residual_layer(1, 14, 14, 1024, 1024, num_repeats=self.num_residual[2]-1)
    self.residual_layer(1, 14, 14, 1024, 2048, stride=2)
    logging.info("conv4 finish!")
    # Conv 5
    self.residual_layer(1, 7, 7, 2048, 2048, num_repeats=self.num_residual[3]-1)
    logging.info("conv5 finish!")

    # Global_Average_Pooling
    # config = [1, 7, 7, 2048, 7, 7]
    # log_file = "kernel_configs/resnext_imagenet_50_avgpool_{}_{}_{}_{}_{}_{}.log".format(*config)
    # self.apply(avgpool, config, log_file)

    # Linear
    config = [1, 1, 1, 2048, 1000]
    log_file = "kernel_configs/resnext_imagenet_50_conv1x1_{}_{}_{}_{}_{}.log".format(*config)
    self.apply(conv1x1, config, log_file)
    logging.info("resnext-imagenet-101 finish!")
  

  def get_total_latency(self):
    logging.info(self.latency_array)
    sum = 0
    for l in self.latency_array:
      sum += l
    return sum
  

  def get_num_of_kernels(self):
    self.num_of_kernels


def run_resnext_imagenet_101(fuse):
  resnext = ResNext(224, [3, 4, 23, 3], fuse=fuse, num_bench=1000, num_repeat=3)
  # resnext.residual_layer(1, 14, 14, 1024, 1024, print_source=False)
  resnext.forward()
  logging.info(resnext.get_total_latency())
  logging.info(resnext.num_of_kernels)


if __name__=="__main__":
  run_resnext_imagenet_101(False)



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

from souffle_model import SouffleModel
from ansor_utils import tune, apply, tvm_bench_func
from resnext_kernels import fused_conv1x1_bn_relu, fused_conv1x1_bn_relu_stride, \
  fused_conv3x3, conv1x1, fused_conv7x7_s2_bn_relu, avgpool, conv3x3, conv1x1_bn_relu, maxpool, \
    fused_conv7x7_s2, batch_norm, relu, hfused_batch_norm, hfused_relu, \
      hfused_conv1x1_stride

folder_path = "ResNextHorizontal_kernel_configs"

class ResNextHorizontal(SouffleModel):
  def __init__(self, image_size, num_residual, fuse=True, tune=False, num_trials=20, num_bench=1, num_repeats=1) -> None:
    self.image_size = image_size
    self.num_residual = num_residual
    self.fuse = fuse
    super().__init__(tune, num_trials, num_bench=num_bench, num_repeats=num_repeats)
  
  
  def residual_layer(self, batch, height, width, \
      in_channels, n_channels, out_channels, stride=1, num_parts=64, print_source=False):
    logging.info("xxxx ({}, {}) input_dim: {}, n_channel: {}, out_dim: {},".format(
            height, width, in_channels, n_channels, out_channels
    ))
    # input shape: (batch, height, width, in_channels)

    # transform_layer
    splitted_channels = n_channels // num_parts

    # horizontal conv1x1
    config = [batch, height, width, in_channels, splitted_channels, num_parts, stride]
    log_file = folder_path+"/resnext_imagenet_50_hfused_conv1x1_stride_{}_{}_{}_{}_{}_{}_s{}.log".format(*config)
    self.run_layer(hfused_conv1x1_stride, config, log_file, print_source=print_source)

    # Note, shape may changed
    height, width = height // stride, width // stride

    # batch_norm + relu
    config = [batch, height, width, splitted_channels, num_parts]
    log_file = folder_path+"/resnext_imagenet_50_hfused_batch_norm_{}_{}_{}_{}_{}.log".format(*config)
    self.run_layer(hfused_batch_norm, config, log_file, print_source=print_source)
    log_file = folder_path+"/resnext_imagenet_50_hfused_relu_{}_{}_{}_{}_{}.log".format(*config)
    self.run_layer(hfused_relu, config, log_file, print_source=print_source)

    # Horizontal conv3x3
    config = [batch, height, width, splitted_channels, splitted_channels, 3, 3, num_parts]
    log_file = folder_path+"/resnext_imagenet_50_fused_conv3x3_{}_{}_{}_{}_{}_{}.log".format(*config)
    self.run_layer(fused_conv3x3, config, log_file, print_source=print_source)
    
    # batch_norm + relu
    config = [batch, height, width, splitted_channels, num_parts]
    log_file = folder_path+"/resnext_imagenet_50_hfused_batch_norm_{}_{}_{}_{}_{}.log".format(*config)
    self.run_layer(hfused_batch_norm, config, log_file, print_source=print_source)
    log_file = folder_path+"/resnext_imagenet_50_hfused_relu_{}_{}_{}_{}_{}.log".format(*config)
    self.run_layer(hfused_relu, config, log_file, print_source=print_source)
    
    # input shape: (batch, height, width, n_channels)

    # transition_layer
    config = [batch, height, width, n_channels, out_channels]
    log_file = folder_path+"/resnext_imagenet_50_conv1x1_{}_{}_{}_{}_{}.log".format(*config)
    self.run_layer(conv1x1, config, log_file, print_source=print_source)
    
    config = [batch, height, width, out_channels]
    log_file = folder_path+"/resnext_imagenet_50_batch_norm_{}_{}_{}_{}.log".format(*config)
    self.run_layer(batch_norm, config, log_file, print_source=print_source)

    # input shape: (batch, height, width, out_channels)

    # last layer of on stage
    if in_channels != out_channels:
      config = [batch, height, width, in_channels, out_channels]
      log_file = folder_path+"/resnext_imagenet_50_conv1x1_{}_{}_{}_{}_{}.log".format(*config)
      self.run_layer(conv1x1, config, log_file)
      config = [batch, height, width, out_channels]
      log_file = folder_path+"/resnext_imagenet_50_batch_norm_{}_{}_{}_{}.log".format(*config)
      self.run_layer(batch_norm, config, log_file, print_source=print_source)
      log_file = folder_path+"/resnext_imagenet_50_relu_{}_{}_{}_{}.log".format(*config)
      self.run_layer(relu, config, log_file, print_source=print_source)


  def forward(self):
    batch_size = 1
    height, width = self.image_size, self.image_size

    # First layer
    config = [batch_size, height, width, 3, 64, 7, 7]
    log_file = folder_path+"/resnext_imagenet_50_fused_conv7x7_s2_{}_{}_{}_{}_{}_{}_{}.log".format(*config)
    self.run_layer(fused_conv7x7_s2, config, log_file)
    height, width = height // 2, width // 2
    config = [batch_size, height, width, 64]
    log_file = folder_path+"/resnext_imagenet_50_batch_norm_{}_{}_{}_{}.log".format(*config)
    self.run_layer(batch_norm, config, log_file)
    log_file = folder_path+"/resnext_imagenet_50_relu_{}_{}_{}_{}.log".format(*config)
    self.run_layer(relu, config, log_file)
    # max_pool
    config = [batch_size, height, width, 64, 3, 3, 1]
    log_file = folder_path+"/resnext_imagenet_50_maxpool_{}_{}_{}_{}_{}_{}_{}.log".format(*config)
    self.run_layer(maxpool, config, log_file)

    height, width = height // 2, width // 2
    # Conv2 (56, 56)
    self.residual_layer(batch_size, height, width, 64, 256, 256, 1, 64)
    for i in range(1, self.num_residual[0]):
      self.residual_layer(batch_size, height, width, 256, 256, 256, 1, 64)
    self.residual_layer(batch_size, height, width, 256, 512, 512, 2, 64)

    height, width = height // 2, width // 2
    # Conv3 (28, 28)
    for i in range(1, self.num_residual[1]):
      self.residual_layer(batch_size, height, width, 512, 512, 512, 1, 64)
    self.residual_layer(batch_size, height, width, 512, 1024, 1024, 1, 64)

    height, width = height // 2, width // 2
    # Conv4 (14, 14)
    for i in range(1, self.num_residual[2]):
      self.residual_layer(batch_size, height, width, 1024, 1024, 1024, 1, 64)
    self.residual_layer(batch_size, height, width, 1024, 2048, 2048, 1, 64)

    height, width = height // 2, width // 2
    # Conv5 (7, 7)
    for i in range(1, self.num_residual[3]):
      self.residual_layer(1, height, width, 2048, 2048, 2048, 1, 64)

    config = [batch_size, height, width, 2048, 7, 7, 1]
    log_file = folder_path+"/resnext_imagenet_50_maxpool_{}_{}_{}_{}_{}_{}_{}.log".format(*config)
    self.run_layer(maxpool, config, log_file)
    # FC
    config = [batch_size, 1, 1, 2048, 2048]
    log_file = folder_path+"/resnext_imagenet_50_conv1x1_{}_{}_{}_{}_{}.log".format(*config)
    self.run_layer(conv1x1, config, log_file)


def run_hfused_resnext():
  model = ResNextHorizontal(224, [3, 4, 23, 3], tune=False, num_trials=1000)
  model.forward()
  print(model.latency_arr)
  print(model.get_total_latency())

if __name__=="__main__":
  run_hfused_resnext()

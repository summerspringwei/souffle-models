
import os, sys
import logging
FORMAT = '%(asctime)s %(filename)s:%(lineno)d %(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO)

print(sys.path)

from souffle_model import SouffleModel
import ansor_utils
from ansor_utils import tune, apply, tvm_bench_func
from efficientnet_pytorch.utils import get_model_params, round_filters, \
  round_repeats, calculate_output_image_size, get_width_and_height_from_size
from efficientnet import conv1x1, batch_norm, depthwise_conv2d, sigmoid, mul, reduce_mean, add, \
  fused_depthwise_conv2d_bn_swish, pointwise_conv2d_bn

folder_path = "VertiFusionEfficientNet_kernel_configs"


class TVMMBConvBlock(SouffleModel):
  def __init__(self, block_args, global_params, image_size=None, batch_size=1, se_mode=0, tune=False, num_trials=20,  num_bench=1000, num_repeats=3):
    super().__init__(tune, num_trials, num_bench=num_bench, num_repeats=num_repeats)
    self._block_args = block_args
    self._bn_mom = 1 - global_params.batch_norm_momentum  # pytorch's difference from tensorflow
    self._bn_eps = global_params.batch_norm_epsilon
    self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
    self.id_skip = block_args.id_skip  # whether to use skip connection and drop connect
    self.num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
    # print("model.py:311 build block: image_size {}, block_args: {}".format(image_size, block_args))

    # Depthwise convolution phase
    self.kernel_size = self._block_args.kernel_size
    self.image_size = image_size
    self.batch_size = batch_size
    # Pointwise convolution phase
    final_oup = self._block_args.output_filters
    self.se_mode = se_mode

  def swish(self, config):
    log_file = folder_path+"/efficient_sigmoid_{}_{}_{}_{}.log".format(*config)
    self.run_layer(sigmoid, config, log_file)
    log_file = folder_path+"/efficient_mul_{}_{}_{}_{}.log".format(*config)
    self.run_layer(mul, config, log_file)

  def forward(self):
    logging.info("({}, {}), inp: {}, oup: {}, final_oup: {} kernel_size: {}, stride: {}".format(
            self.image_size, self.image_size, self._block_args.input_filters, self._block_args.input_filters * self._block_args.expand_ratio, 
            self._block_args.output_filters,self._block_args.kernel_size, self._block_args.stride
    ))
    inp = self._block_args.input_filters  # number of input channels
    oup = self._block_args.input_filters * self._block_args.expand_ratio  # number of output channels
    final_oup = self._block_args.output_filters
    batch_size = self.batch_size
    height, width = get_width_and_height_from_size(self.image_size)
    
    # Expansion and Depthwise Convolution
    if self._block_args.expand_ratio != 1:
      config = [batch_size, height, width, inp, oup]
      log_file = folder_path+"/efficient_conv1x1_{}_{}_{}_{}_{}.log".format(*config)
      self.run_layer(conv1x1, config, log_file)
      config = [batch_size, height, width, oup]
      log_file = folder_path+"/efficient_batch_norm_{}_{}_{}_{}.log".format(*config)
      self.run_layer(batch_norm, config, log_file)

      self.swish(config)
    
    # Depthwise convolution phase
    s = self._block_args.stride if isinstance(self._block_args.stride, int)  else self._block_args.stride[0]
    config = [height, width, oup, self.kernel_size, self.kernel_size, s, batch_size]
    log_file = folder_path+"/efficient_depthwise_conv2d_{}_{}_{}_{}_{}_s{}_{}.log".format(*config)
    self.run_layer(depthwise_conv2d, config, log_file)
    # Note, shape may change here
    height, width = height // s, width // s
    config = [batch_size, height, width, oup]
    log_file = folder_path+"/efficient_batch_norm_{}_{}_{}_{}.log".format(*config)
    self.run_layer(batch_norm, config, log_file)
    self.swish(config)

    # Squeeze and Excitation layer, if desired
    if self.has_se:
      h, w = self.image_size
      logging.info("se module h {} w {} in_channel: {} out_channel: {}".format(self.image_size, self.image_size, oup, self.num_squeezed_channels))
      config = [h, w, oup]
      log_file = folder_path+"/efficient_reduce_mean_{}_{}_{}.log".format(*config)
      self.run_layer(reduce_mean, config, log_file)
      config = [batch_size, 1, 1, oup, self.num_squeezed_channels]
      log_file = folder_path+"/efficient_conv1x1_{}_{}_{}_{}_{}.log".format(*config)
      self.run_layer(conv1x1, config, log_file)
      config = [batch_size, 1, 1, self.num_squeezed_channels]
      self.swish(config)
      config = [batch_size, 1, 1, self.num_squeezed_channels, oup]
      log_file = folder_path+"/efficient_conv1x1_{}_{}_{}_{}_{}.log".format(*config)
      self.run_layer(conv1x1, config, log_file)
      # TODO
      config = [batch_size, 1, 1, oup]
      self.swish(config)

    # Pointwise Convolution
    config = [batch_size, height, width, oup, final_oup]
    log_file = folder_path+"/efficient_conv1x1_{}_{}_{}_{}_{}.log".format(*config)
    self.run_layer(conv1x1, config, log_file)
    config = [batch_size, height, width, oup]
    log_file = folder_path+"/efficient_batch_norm_{}_{}_{}_{}.log".format(*config)
    self.run_layer(batch_norm, config, log_file)

    input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
    if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
      config = [batch_size, height, width, oup]
      log_file = folder_path+"/efficient_add_{}_{}_{}_{}.log".format(*config)
      self.run_layer(add, config, log_file)


class NaiveTVMEfficientNet(SouffleModel):
  def __init__(self, model_name, se_mode=0, tune=False, num_trials=20, num_bench=1000, num_repeats=3):
    super().__init__(tune, num_trials, num_bench=num_bench, num_repeats=num_repeats)
    blocks_args, global_params = get_model_params(model_name, None)
    assert isinstance(blocks_args, list), 'blocks_args should be a list'
    assert len(blocks_args) > 0, 'block args must be greater than 0'
    self._global_params = global_params
    self._blocks_args = blocks_args
    print("block_args{}".format(self._blocks_args))
    print("global_params{}".format(self._global_params))

    # Get stem static or dynamic convolution depending on image size
    image_size = global_params.image_size

    # Stem
    image_size = calculate_output_image_size(image_size, 2)

    # Build blocks
    self._blocks = []
    for block_args in self._blocks_args:
      print(block_args, block_args.num_repeat)
      # Update block input and output filters based on depth multiplier.
      block_args = block_args._replace(
          input_filters=round_filters(block_args.input_filters, self._global_params),
          output_filters=round_filters(block_args.output_filters, self._global_params),
          num_repeat=round_repeats(block_args.num_repeat, self._global_params)
      )
      
      # The first block needs to take care of stride and filter size increase.
      self._blocks.append(TVMMBConvBlock(block_args, self._global_params, \
        image_size=image_size, se_mode=se_mode, tune=self.tune, num_trials=self.num_trials, \
          num_bench=self.num_bench, num_repeats=self.num_repeats))
      image_size = calculate_output_image_size(image_size, block_args.stride)
      if block_args.num_repeat > 1:  # modify block_args to keep same output size
          block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
      for _ in range(block_args.num_repeat - 1):
          self._blocks.append(TVMMBConvBlock(block_args, self._global_params, \
            image_size=image_size, se_mode=se_mode, tune=self.tune, num_trials=self.num_trials, \
              num_bench=self.num_bench, num_repeats=self.num_repeats))
  

  def forward(self):
    # Head
    config = [224, 224, 32, 3, 3, 2, 1]
    log_file = "kernel_configs/efficient_depthwise_conv2d_bn_swish_{}_{}_{}_{}_{}_{}_{}.log".format(*config)
    self.run_layer(fused_depthwise_conv2d_bn_swish, config, log_file)
    
    for blk in self._blocks:
      blk.forward()
      self.latency_arr.append(blk.get_total_latency())
      self.num_of_kernels += blk.num_of_kernels

    config = [7, 7, 320, 1280, 1]
    log_file = "kernel_configs/efficient_pointwise_conv2d_bn_{}_{}_{}_{}_{}.log".format(*config)
    self.run_layer(pointwise_conv2d_bn, config, log_file)
    config = [1, 1, 1280, 1000, 1]
    log_file = "kernel_configs/efficient_pointwise_conv2d_bn_{}_{}_{}_{}_{}.log".format(*config)
    self.run_layer(pointwise_conv2d_bn, config, log_file)


def run_EfficientNet():
  model = NaiveTVMEfficientNet("efficientnet-b0", se_mode=0, tune=False, num_trials=400)
  model.forward()
  print(model.latency_arr)
  print("Total latency {}".format(model.get_total_latency()))
  print(model.num_of_kernels)


if __name__=="__main__":
  run_EfficientNet()

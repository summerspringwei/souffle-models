import torch

import efficientnet_se_module_binding

def test_se_module(batch_size, height, width, in_channel, reduce_channel, opt_level=1):
  input_tensor = torch.ones((batch_size, in_channel, height, width), dtype=torch.float32, device="cuda").to("cuda")
  se_reduce_weight = torch.ones((reduce_channel, in_channel), dtype=torch.float32, device="cuda").to("cuda")
  se_expand_weight = torch.ones((in_channel, reduce_channel), dtype=torch.float32, device="cuda").to("cuda")
  se_short_cut_add = efficientnet_se_module_binding.torch_dispatch_efficientnet_se_module_v2_short_cut_fused(input_tensor, se_reduce_weight, se_expand_weight, opt_level)
  print(se_short_cut_add)


def test_all_with_opt_level(opt_level):
  test_se_module(1, 112, 112, 32, 8, opt_level)
  test_se_module(1, 56, 56, 96, 4, opt_level)
  test_se_module(1, 56, 56, 144, 6, opt_level)
  test_se_module(1, 28, 28, 144, 6, opt_level)
  test_se_module(1, 28, 28, 240, 10, opt_level)
  test_se_module(1, 14, 14, 240, 10, opt_level)
  test_se_module(1, 14, 14, 480, 20, opt_level)
  test_se_module(1, 14, 14, 672, 28, opt_level)
  test_se_module(1, 7, 7, 672, 28, opt_level)
  test_se_module(1, 7, 7, 1152, 48, opt_level)


if __name__ == "__main__":
  test_all_with_opt_level(0)
  test_all_with_opt_level(1)
  test_all_with_opt_level(2)

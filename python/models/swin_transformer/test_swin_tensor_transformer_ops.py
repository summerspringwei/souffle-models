import tvm
from tvm import te, tir, auto_scheduler, topi, autotvm
import numpy as np
import torch

from ...ansor_utils import tune, apply

from swin_self_attention import roll, window_partition, \
  fused_roll_window_partition, fused_reshape_permute, \
  single_patch_merging, fused_patch_merging_reshape, \
  fused_patch_merging_reshape_reduce_sum, layer_normalization_variance, layer_normalization_normal, \
  fused_window_reverse_roll_add


def test_roll(batch_size, height, width, channel, shift_size):
    a_np = np.random.random(
        (batch_size, height, width, channel)).astype(np.float32)
    b_np = np.zeros((batch_size, height, width, channel)).astype(np.float32)
    x, shifted_x = roll(batch_size, height, width, channel, shift_size,
                        "float32")
    s = te.create_schedule(shifted_x.op)
    func = tvm.build(s, [x, shifted_x], "llvm")
    a_tvm = tvm.nd.array(a_np)
    b_tvm = tvm.nd.array(b_np)
    func(a_tvm, b_tvm)
    print(b_tvm)
    b_torch = torch.roll(
        torch.tensor(a_np).float(), (shift_size, shift_size), (1, 2)).numpy()
    np.testing.assert_allclose(b_torch, b_tvm.numpy())


def torch_window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    (B, H, W, C) = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size,
               C)
    windows = x.permute(0, 1, 3, 2, 4,
                        5).contiguous().view(-1, window_size, window_size, C)
    # windows = x.permute(0, 1, 3, 2, 4, 5)
    return windows


def torch_window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W))
    x = windows.view(B, H // window_size, W // window_size, window_size,
                     window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def test_window_partition(batch_size,
                          height,
                          width,
                          channel,
                          window_size,
                          dtype="float32"):
    x_shape = (batch_size, height, width, channel)
    a_np = np.random.random(x_shape).astype(np.float32)
    # windows_shape = (batch_size, height//window_size, width//window_size, window_size, window_size, channel)
    windows_shape = (batch_size * (height // window_size) *
                     (width // window_size), window_size, window_size, channel)
    b_np = np.zeros(windows_shape).astype(np.float32)
    x, x_windows = window_partition(batch_size, height, width, channel,
                                    window_size, dtype)
    s = te.create_schedule(x_windows.op)
    func = tvm.build(s, [x, x_windows], "llvm")
    a_tvm = tvm.nd.array(a_np)
    b_tvm = tvm.nd.array(b_np)
    func(a_tvm, b_tvm)
    print(b_tvm)
    b_torch = torch_window_partition(
        torch.tensor(a_np).to(torch.float32), window_size)
    np.testing.assert_allclose(b_torch, b_tvm.numpy())


def test_fused_roll_window_partition(batch_size,
                                     height,
                                     width,
                                     channel,
                                     shift_size,
                                     window_size,
                                     dtype="float32"):
    a_shape = (batch_size, height, width, channel)
    a_np = np.random.random(a_shape).astype(np.float32)
    b_np = np.zeros(
        (batch_size * (height // window_size) * (width // window_size),
         window_size, window_size, channel)).astype(np.float32)
    x, x_window_partition = fused_roll_window_partition(batch_size,
                                                        height,
                                                        width,
                                                        channel,
                                                        shift_size,
                                                        window_size,
                                                        dtype="float32")
    s = te.create_schedule(x_window_partition.op)
    func = tvm.build(s, [x, x_window_partition], "llvm")
    a_tvm = tvm.nd.array(a_np)
    b_tvm = tvm.nd.array(b_np)
    func(a_tvm, b_tvm)
    print(b_tvm)
    shift_b = torch.roll(
        torch.tensor(a_np).float(), (shift_size, shift_size), (1, 2))
    b_torch = torch_window_partition(shift_b, window_size)
    np.testing.assert_allclose(b_torch, b_tvm.numpy())


def test_fused_reshape_permute(batch_size,
                               height,
                               width,
                               channel,
                               window_size,
                               num_heads,
                               dtype="float16"):
    x_shape = (batch_size * height * width, 3 * channel)
    num_height = height // window_size
    num_width = width // window_size
    seq_length = channel // num_heads
    x_permuted_shape = (3, batch_size * num_height * num_width, num_heads,
                        window_size * window_size, seq_length)
    # Get tvm result
    x, x_permuted = fused_reshape_permute(batch_size, height, width, channel,
                                          window_size, num_heads, dtype)
    s = te.create_schedule([x_permuted.op])
    func = tvm.build(s, [x, x_permuted], "llvm")
    dev = tvm.cpu(0)
    x_np = np.random.random(x_shape).astype(np.float32)
    x_permuted = np.zeros(x_permuted_shape).astype(np.float32)
    x_tvm = tvm.nd.array(x_np, dev)
    x_permuted_tvm = tvm.nd.array(x_permuted, dev)
    func(x_tvm, x_permuted_tvm)
    # Get torch result
    x_reshaped_torch = torch.reshape(\
      torch.tensor(x_np, dtype=torch.float32), \
        (batch_size, num_height, window_size, num_width, window_size, 3, num_heads, seq_length))
    x_permuted_torch = torch.permute(x_reshaped_torch,
                                     (5, 0, 1, 3, 6, 2, 4, 7))
    x_reshaped_permuted_torch = torch.reshape(x_permuted_torch,
                                              x_permuted_shape)
    np.testing.assert_allclose(
        x_permuted_tvm.numpy().astype(np.float32),
        x_reshaped_permuted_torch.numpy().astype(np.float32))


def torch_patch_merging(x, batch_size, height, width, channel):
    """Copy From swin-transformer
  """
    x = torch.tensor(x)
    x = x.view(batch_size, height, width, channel)
    x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
    x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
    x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
    x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
    x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
    return x


def torch_patch_merging_reshape(x, batch_size, height, width, channel):
    """Copy From swin-transformer
  """
    x = torch.tensor(x)
    x = x.view(batch_size, height, width, channel)
    x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
    x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
    x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
    x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
    x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
    x = x.view(batch_size * height // 2 * width // 2, 4 * channel)
    return x


def test_patch_mering(batch_size, height, width, channel):
    x_np = np.random.random(
        (batch_size, height, width, channel)).astype(np.float32)
    x_patched_np = np.zeros(
        (batch_size, height // 2, width // 2, 4 * channel)).astype(np.float32)
    patched_torch = torch_patch_merging(x_np, batch_size, height, width,
                                        channel)
    x, x_patched = single_patch_merging(batch_size, height, width, channel,
                                        "float32")
    s = te.create_schedule([x_patched.op])
    func = tvm.build(s, [x, x_patched], "llvm")
    dev = tvm.cpu(0)
    x_tvm = tvm.nd.array(x_np, dev)
    x_patched_tvm = tvm.nd.array(x_patched_np, dev)
    func(x_tvm, x_patched_tvm)
    np.testing.assert_allclose(x_patched_tvm.numpy(), patched_torch.numpy())


def test_patch_mering_reshape(batch_size, height, width, channel):
    x_np = np.random.random(
        (batch_size, height, width, channel)).astype(np.float32)
    x_patched_np = np.zeros((batch_size * height // 2 * width // 2,
                             4 * channel)).astype(np.float32)
    patched_torch = torch_patch_merging_reshape(x_np, batch_size, height,
                                                width, channel)
    x, x_patched = fused_patch_merging_reshape(batch_size, height, width,
                                               channel, "float32")
    s = te.create_schedule([x_patched.op])
    func = tvm.build(s, [x, x_patched], "llvm")
    dev = tvm.cpu(0)
    x_tvm = tvm.nd.array(x_np, dev)
    x_patched_tvm = tvm.nd.array(x_patched_np, dev)
    func(x_tvm, x_patched_tvm)
    np.testing.assert_allclose(x_patched_tvm.numpy(), patched_torch.numpy())


def test_patch_mering_reshape_layer_norm(batch_size, height, width, channel):
    # Declare numpy array
    x_reduced_shape = (batch_size * height // 2 * width // 2, )
    x_shape = (batch_size * height // 2 * width // 2, 4 * channel)
    import PIL
    from PIL import Image
    img = Image.open("/home/xiachunwei/Software/tensor-compiler/img/kenan.jpg")
    img = img.resize((height, width))
    x_np = np.array(img).reshape(
        (batch_size, height, width, channel)).astype(np.float32)
    # x_np = np.random.random((batch_size, height, width, channel)).astype(np.float32)
    # x_np = np.array([[1,2,3], [4,5,6],[1,2,3], [4,5,6]]).reshape((1,2,2,3)).astype(np.float32)
    x_patched_np = np.zeros(x_shape).astype(np.float32)
    x_reduce_sum_np = np.zeros(x_reduced_shape).astype(np.float32)
    x_mean_np = np.zeros(x_reduced_shape).astype(np.float32)
    x_variance_np = np.zeros(x_reduced_shape).astype(np.float32)
    x_output_np = np.zeros(x_shape).astype(np.float32)
    # Build tvm func
    x, x_patched, x_reduced_sum = fused_patch_merging_reshape_reduce_sum(
        batch_size, height, width, channel, "float32")
    s_fused_patch_merging_reshape_reduce_sum = te.create_schedule(
        [x_reduced_sum.op])
    func_fused_patch_merging_reshape_reduce_sum = tvm.build(
        s_fused_patch_merging_reshape_reduce_sum,
        [x, x_patched, x_reduced_sum], "llvm")
    x_merged, x_reduce_sum, x_mean, x_variance = layer_normalization_variance(
        batch_size, height, width, channel, "float32")
    s_layer_normalization_variance = te.create_schedule([x_variance.op])
    func_layer_normalization_variance = tvm.build(
        s_layer_normalization_variance,
        [x_merged, x_reduce_sum, x_mean, x_variance], "llvm")
    x, x_mean, x_variance, x_output = layer_normalization_normal(
        batch_size, height, width, channel, 1.0, 0.0, "float32")
    s_layer_normalization_normal = te.create_schedule([x_output.op])
    func_layer_normalization_normal = tvm.build(
        s_layer_normalization_normal, [x, x_mean, x_variance, x_output],
        "llvm")
    # Declare tvm nd array
    dev = tvm.cpu(0)
    x_tvm = tvm.nd.array(x_np, dev)
    x_patched_tvm = tvm.nd.array(x_patched_np, dev)
    x_reduce_sum_tvm = tvm.nd.array(x_reduce_sum_np, dev)
    x_mean_tvm = tvm.nd.array(x_mean_np, dev)
    x_variance_tvm = tvm.nd.array(x_variance_np, dev)
    x_output_tvm = tvm.nd.array(x_output_np, dev)
    # Run tvm func
    func_fused_patch_merging_reshape_reduce_sum(x_tvm, x_patched_tvm,
                                                x_reduce_sum_tvm)
    func_layer_normalization_variance(x_patched_tvm, x_reduce_sum_tvm,
                                      x_mean_tvm, x_variance_tvm)
    func_layer_normalization_normal(x_patched_tvm, x_mean_tvm, x_variance_tvm,
                                    x_output_tvm)
    # PyTorch version
    with torch.no_grad():
        patched_torch = torch_patch_merging_reshape(x_np, batch_size, height,
                                                    width, channel)
        normalized_torch = torch.nn.LayerNorm(
            4 * channel, dtype=torch.float32)(patched_torch)
    # Verify
    np.testing.assert_allclose(normalized_torch.numpy(), x_output_tvm.numpy())


def test_patch_mering_reshape_layer_norm_float64(batch_size, height, width,
                                                 channel):
    # Declare numpy array
    x_reduced_shape = (batch_size * height // 2 * width // 2, )
    x_shape = (batch_size * height // 2 * width // 2, 4 * channel)
    import PIL
    from PIL import Image
    img = Image.open("/home/xiachunwei/Software/tensor-compiler/img/kenan.jpg")
    img = img.resize((height, width))
    x_np = np.array(img).reshape(
        (batch_size, height, width, channel)).astype(np.float64)
    # x_np = np.random.random((batch_size, height, width, channel)).astype(np.float64)
    # x_np = np.array([[1,2,3], [4,5,6],[1,2,3], [4,5,6]]).reshape((1,2,2,3)).astype(np.float64)
    x_patched_np = np.zeros(x_shape).astype(np.float64)
    x_reduce_sum_np = np.zeros(x_reduced_shape).astype(np.float64)
    x_mean_np = np.zeros(x_reduced_shape).astype(np.float64)
    x_variance_np = np.zeros(x_reduced_shape).astype(np.float64)
    x_output_np = np.zeros(x_shape).astype(np.float64)
    # Build tvm func
    x, x_patched, x_reduced_sum = fused_patch_merging_reshape_reduce_sum(
        batch_size, height, width, channel, "float64")
    s_fused_patch_merging_reshape_reduce_sum = te.create_schedule(
        [x_reduced_sum.op])
    func_fused_patch_merging_reshape_reduce_sum = tvm.build(
        s_fused_patch_merging_reshape_reduce_sum,
        [x, x_patched, x_reduced_sum], "llvm")
    x_merged, x_reduce_sum, x_mean, x_variance = layer_normalization_variance(
        batch_size, height, width, channel, "float64")
    s_layer_normalization_variance = te.create_schedule([x_variance.op])
    func_layer_normalization_variance = tvm.build(
        s_layer_normalization_variance,
        [x_merged, x_reduce_sum, x_mean, x_variance], "llvm")
    x, x_mean, x_variance, x_output = layer_normalization_normal(
        batch_size, height, width, channel, 1.0, 0.0, "float64")
    s_layer_normalization_normal = te.create_schedule([x_output.op])
    func_layer_normalization_normal = tvm.build(
        s_layer_normalization_normal, [x, x_mean, x_variance, x_output],
        "llvm")
    # Declare tvm nd array
    dev = tvm.cpu(0)
    x_tvm = tvm.nd.array(x_np, dev)
    x_patched_tvm = tvm.nd.array(x_patched_np, dev)
    x_reduce_sum_tvm = tvm.nd.array(x_reduce_sum_np, dev)
    x_mean_tvm = tvm.nd.array(x_mean_np, dev)
    x_variance_tvm = tvm.nd.array(x_variance_np, dev)
    x_output_tvm = tvm.nd.array(x_output_np, dev)
    # Run tvm func
    func_fused_patch_merging_reshape_reduce_sum(x_tvm, x_patched_tvm,
                                                x_reduce_sum_tvm)
    func_layer_normalization_variance(x_patched_tvm, x_reduce_sum_tvm,
                                      x_mean_tvm, x_variance_tvm)
    func_layer_normalization_normal(x_patched_tvm, x_mean_tvm, x_variance_tvm,
                                    x_output_tvm)
    # PyTorch version
    with torch.no_grad():
        patched_torch = torch_patch_merging_reshape(x_np, batch_size, height,
                                                    width, channel)
        normalized_torch = torch.nn.LayerNorm(
            4 * channel, dtype=torch.float64)(patched_torch)
    # Verify
    np.testing.assert_allclose(normalized_torch.numpy(),
                               x_output_tvm.numpy(),
                               rtol=0.0001)


def test_layer_norm():
    a = [[1, 2, 3], [4, 5, 6]]
    with torch.no_grad():
        print(
            torch.nn.LayerNorm(3)(torch.tensor(
                np.array(a).reshape((2, 3)).astype(np.float32))))


def test_fused_window_reverse_roll_add(batch_size,
                                       height,
                                       width,
                                       channel,
                                       shift_size,
                                       window_size,
                                       dtype="float32"):
    args = fused_window_reverse_roll_add(batch_size, height, width, channel,
                                         shift_size, window_size, dtype)
    # By default the last element in args is output
    s = te.create_schedule([args[-1].op])
    func = tvm.build(s, args, "llvm")

    # Create tvm nd array
    dev = tvm.cpu(0)
    arr_tvm = []
    arr_torch = []
    for arg in args:
        shape_int_array = []
        for imm in (arg.shape):
            shape_int_array.append(imm.__int__())
        arg_np = np.random.rand(*(shape_int_array)).astype(arg.dtype)
        arr_tvm.append(tvm.nd.array(arg_np, dev))
        arr_torch.append(torch.tensor(arg_np).cpu())
    # Run tvm func
    func(*arr_tvm)
    # Get torch tensor
    torch_x, torch_short_cut, _ = arr_torch
    torch_permute = torch_window_reverse(torch_x, window_size, height, width)
    torch_roll = torch.roll(torch_permute, (shift_size, shift_size), (1, 2))
    torch_output = torch_roll + torch_short_cut
    np.testing.assert_allclose(arr_tvm[-1].numpy(), torch_output.numpy())


if __name__ == "__main__":
    # test_patch_mering(1, 56, 56, 128)
    # test_patch_mering_reshape(1, 56, 56, 128)
    # test_patch_mering_reshape_layer_norm(1, 56, 56, 3)
    # test_patch_mering_reshape_layer_norm_float64(1, 56, 56, 3)
    # test_patch_mering_reshape_layer_norm(1,2,2,3)
    # test_layer_norm()
    test_fused_window_reverse_roll_add(1, 56, 56, 128, 3, 7)

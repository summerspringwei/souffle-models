"""This file Implemente the Mlp module in swin-transformer, which is part of FFN module

The correspinding PyTorch code:
  def forward(self, x):
      x = self.fc1(x) # Dense matmul with weight (channel, 4*channel)
      x = self.act(x) # GELU activation
      x = self.drop(x)# dropout is optimized out in inference
      x = self.fc2(x) # Dense matmul with weight (4*channel, channel)
      x = self.drop(x)
      return x
  x = x + self.drop_path(self.mlp(self.norm2(x))) # self.norm2 is LayerNorm

`fused_layer_norm_matmul_tensorcore_gelu` implements self.norm2+self.fc1(x)+self.act(x)
`fused_matmul_tensorcore_add` implements x+self.fc2(x)
"""

import math, os, sys, pathlib

sys.path.append("/home/xiachunwei/Software/clean_tvm/tvm/python")
import tvm
from tvm import te, tir, topi, autotvm, auto_scheduler
from tvm.topi.cuda import tag
from tvm.topi.utils import traverse_inline, get_const_tuple

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.sep + "../../")
import ansor_utils
from ansor_utils import tune, apply, tvm_bench_func, autotvm_tune
from parse_tvm_schedule import parseDim
import swin_transformer_binding

@auto_scheduler.register_workload
def relu(batch, height, width, channels):
    in_out_shape = (batch, height, width, channels)
    input_tensor = te.placeholder(in_out_shape, "float32", name="input_tensor")
    # Relu
    output = te.compute(in_out_shape,\
      lambda b, h, w, o: tir.max(input_tensor[b, h, w, o], 0))

    return [input_tensor, output]


@autotvm.register_topi_compute("fused_layer_norm_matmul_tensorcore_gelu.cuda")
def fused_layer_norm_matmul_tensorcore_gelu(cfg,
                                            x,
                                            x_mean,
                                            x_variance_sum,
                                            weight,
                                            scale,
                                            gamma=1,
                                            beta=0,
                                            dtype="float16"):
    """Dense tensorcore operator on CUDA"""
    matmul = _fused_layer_norm_matmul_tensorcore_gelu(x, x_mean,
                                                      x_variance_sum, weight,
                                                      scale, gamma, beta,
                                                      dtype)
    return matmul


@autotvm.register_topi_schedule("fused_layer_norm_matmul_tensorcore_gelu.cuda")
def schedule_fused_layer_norm_matmul_tensorcore_gelu(cfg, outs):
    """Schedule dense operator using Tensorcore"""
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])
    from inline_matmul_utils import _schedule_fused_precedence_dense_tensorcore

    def _callback(op):
        if op.tag == "dense_tensorcore":
            _schedule_fused_precedence_dense_tensorcore(cfg, s, op.output(0))

    traverse_inline(s, outs[0].op, _callback)
    return s


def _fused_layer_norm_matmul_tensorcore_gelu(x,
                                             x_mean,
                                             x_variance_sum,
                                             weight,
                                             scale,
                                             gamma=1,
                                             beta=0,
                                             dtype="float16"):
    x_normalized = te.compute(
        x.shape,
        lambda i, j: ((x[i, j] - x_mean[i]) /
                      (x_variance_sum[i] * scale + 1e5)) * gamma + beta,
        name="x_normalized",
        tag="elemwise").astype(dtype)
    matmul = topi.cuda.dense_tensorcore_cuda(x_normalized, weight)
    gelu = te.compute(
        matmul.shape,
        lambda i, j: 0.5 * matmul[i, j] * (1 + tir.tanh(
            math.sqrt(2 / math.pi) *
            (matmul[i, j] + 0.044715 * tir.power(matmul[i, j], 3)))),
        name="gelu",
        tag="elemwise").astype(dtype)
    return gelu


def auto_tvm_tune_fused_layer_norm_matmul_tensorcore_gelu(
        batch_size,
        height,
        width,
        in_dim,
        out_dim,
        model_name="swin_transform",
        dtype="float16"):
    """Layer Norm: Compute the normalize
  Dense: (BHW, 4C) x (4C, 2C) -> (BHW, 2C)
  """
    log_file = "kernel_configs/{}_auto_tvm_tune_fused_layer_norm_matmul_tensorcore_gelu_{}_{}_{}_{}_{}.log".format(
        model_name, batch_size, height, width, in_dim, out_dim)
    x_shape = (batch_size * height * width, in_dim)
    reduced_shape = (batch_size * height * width, )
    scale = 1.0 / in_dim

    x = te.placeholder(x_shape, dtype, name="x")
    x_mean = te.placeholder(reduced_shape, dtype, name="x_mean")
    x_variance_sum = te.placeholder(reduced_shape,
                                    dtype,
                                    name="x_variance_sum")
    weight = te.placeholder((out_dim, in_dim), dtype, name="weight")

    task = tvm.autotvm.task.create("fused_layer_norm_matmul_tensorcore_gelu.cuda", \
        args=(x, x_mean, x_variance_sum, weight, scale), target='cuda')
    print(task.config_space)
    measure_option = autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.LocalRunner(repeat=3,
                                   number=1000,
                                   min_repeat_ms=10,
                                   timeout=4),
    )
    tuner = autotvm.tuner.XGBTuner(task)
    tuner.tune(
        n_trial=2000,
        measure_option=measure_option,
        callbacks=[autotvm.callback.log_to_file(log_file)],
    )


def auto_tvm_apply_fused_layer_norm_matmul_tensorcore_gelu(
        batch_size,
        height,
        width,
        in_dim,
        out_dim,
        model_name="swin_transform",
        dtype="float16",
        num_bench=1000):
    log_file = "kernel_configs/{}_auto_tvm_tune_fused_layer_norm_matmul_tensorcore_gelu_{}_{}_{}_{}_{}.log".format(
        model_name, batch_size, height, width, in_dim, out_dim)

    x_shape = (batch_size * height * width, in_dim)
    reduced_shape = (batch_size * height * width, )
    scale = 1.0 / in_dim

    x = te.placeholder(x_shape, dtype, name="x")
    x_mean = te.placeholder(reduced_shape, dtype, name="x_mean")
    x_variance_sum = te.placeholder(reduced_shape,
                                    dtype,
                                    name="x_variance_sum")
    weight = te.placeholder((out_dim, in_dim), dtype, name="weight")
    func_name =  pathlib.Path(log_file).stem
    with autotvm.apply_history_best(log_file):
        with tvm.target.Target("cuda"):
            out = fused_layer_norm_matmul_tensorcore_gelu(
                x, x_mean, x_variance_sum, weight, scale)
            s = schedule_fused_layer_norm_matmul_tensorcore_gelu(out)
            args = [x, x_mean, x_variance_sum, weight, out]

            func = tvm.build(s, args, name=func_name)
            str_schedule = str(tvm.lower(s, args, simple_mode=True))
            str_cuda_source = str(func.imported_modules[0].get_source())
            # str_cuda_source = str(func.imported_modules[0].get_ptx_source())
            # print(tvm.lower(s, args, simple_mode=True))
            # print(func.imported_modules[0].get_source())
            # parseDim(str_schedule)

    dev = tvm.cuda(0)
    return tvm_bench_func(func, args, dev,
                          num_bench=num_bench), (str_schedule, str_cuda_source)


@autotvm.register_topi_compute("fused_matmul_tensorcore_add.cuda")
def fused_matmul_tensorcore_add(cfg, x, short_cut, weight, dtype="float16"):
    """Dense tensorcore operator on CUDA"""
    matmul = _fused_matmul_tensorcore_add(x, short_cut, weight, dtype)
    return matmul


@autotvm.register_topi_schedule("fused_matmul_tensorcore_add.cuda")
def schedule_fused_matmul_tensorcore_add(cfg, outs):
    """Schedule dense operator using Tensorcore"""
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])
    from inline_matmul_utils import _schedule_fused_precedence_dense_tensorcore

    def _callback(op):
        if op.tag == "dense_tensorcore":
            _schedule_fused_precedence_dense_tensorcore(cfg, s, op.output(0))

    traverse_inline(s, outs[0].op, _callback)
    return s


def _fused_matmul_tensorcore_add(x, short_cut, weight, dtype="float16"):
    matmul = topi.cuda.dense_tensorcore_cuda(x, weight)
    add = te.compute(matmul.shape,
                     lambda i, j: matmul[i, j] + short_cut[i, j],
                     name="add",
                     tag="elemwise")
    return add


def auto_tvm_tune_fused_matmul_tensorcore_add(batch_size,
                                              height,
                                              width,
                                              in_dim,
                                              out_dim,
                                              model_name="swin_transform",
                                              dtype="float16"):
    """Layer Norm: Compute the normalize
  Dense: (BHW, 4C) x (4C, 2C) -> (BHW, 2C)
  """
    log_file = "kernel_configs/{}_auto_tvm_tune_fused_matmul_tensorcore_add_{}_{}_{}_{}_{}.log".format(
        model_name, batch_size, height, width, in_dim, out_dim)

    x_shape = (batch_size * height * width, in_dim)

    x = te.placeholder(x_shape, dtype, name="x")
    short_cut = te.placeholder(x_shape, dtype, name="short_cut")
    weight = te.placeholder((out_dim, in_dim), dtype, name="weight")

    task = tvm.autotvm.task.create("fused_matmul_tensorcore_add.cuda", \
        args=(x, short_cut, weight), target='cuda')
    # print(task.config_space)
    measure_option = autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.LocalRunner(repeat=3,
                                   number=1000,
                                   min_repeat_ms=10,
                                   timeout=4),
    )
    tuner = autotvm.tuner.XGBTuner(task)
    tuner.tune(
        n_trial=2000,
        measure_option=measure_option,
        callbacks=[autotvm.callback.log_to_file(log_file)],
    )


def auto_tvm_apply_fused_matmul_tensorcore_add(batch_size,
                                               height,
                                               width,
                                               in_dim,
                                               out_dim,
                                               model_name="swin_transform",
                                               dtype="float16",
                                               num_bench=1000):
    log_file = "kernel_configs/{}_auto_tvm_tune_fused_matmul_tensorcore_add_{}_{}_{}_{}_{}.log".format(
        model_name, batch_size, height, width, in_dim, out_dim)

    x_shape = (batch_size * height * width, in_dim)

    x = te.placeholder(x_shape, dtype, name="x")
    short_cut = te.placeholder(x_shape, dtype, name="short_cut")
    weight = te.placeholder((out_dim, in_dim), dtype, name="weight")
    func_name =  pathlib.Path(log_file).stem
    with autotvm.apply_history_best(log_file):
        with tvm.target.Target("cuda"):
            out = fused_matmul_tensorcore_add(x, short_cut, weight)
            s = schedule_fused_matmul_tensorcore_add(out)
            args = [x, short_cut, weight, out]

            func = tvm.build(s, args, name=func_name)
            str_schedule = str(tvm.lower(s, args, simple_mode=True))
            str_cuda_source = str(func.imported_modules[0].get_source())
            # str_cuda_source = str(func.imported_modules[0].get_ptx_source())
            # print(tvm.lower(s, args, simple_mode=True))
            # print(func.imported_modules[0].get_source())
            parseDim(str_schedule)

    dev = tvm.cuda(0)
    return tvm_bench_func(func, args, dev,
                          num_bench=num_bench), (str_schedule, str_cuda_source)


def autotvm_tune_dense(M, N, K, log_file, num_trials, dtype="float16"):
    A = te.placeholder((M, K), dtype)
    B = te.placeholder((N, K), dtype)
    task = tvm.autotvm.task.create("dense_tensorcore.cuda", \
      args=(A, B), target='cuda')
    autotvm_tune(task, log_file, num_trials)


def autotvm_apply_dense(M, N, K, log_file, dtype="float16"):
    A = te.placeholder((M, K), dtype)
    B = te.placeholder((N, K), dtype)
    with autotvm.apply_history_best(log_file):
        with tvm.target.Target("cuda"):
            out = topi.cuda.dense_tensorcore(A, B, out_dtype=dtype)
            s = topi.cuda.schedule_dense_tensorcore(out)
            args = [A, B, out]
            func = tvm.build(s, args, name="auto_tvm_dense_{}_{}_{}".format(M, N, K))
            func.export_library(log_file + ".so")
    dev = tvm.cuda(0)

    return tvm_bench_func(func, args, dev)


def fused_layer_normalization_tensorcore_gelu_tune(tunning=False):
    kernel_configs = [
        # [1,64,64,128,512, "swin_transform", "float16"],
        # [1,32,32,256,1024, "swin_transform", "float16"],
        [1, 16, 16, 512, 2048, "swin_transform", "float16"],
        # [1,8,8,1024,4096, "swin_transform", "float16"],
    ]
    for config in kernel_configs:
        if tunning:
            auto_tvm_tune_fused_layer_norm_matmul_tensorcore_gelu(*config)
        auto_tvm_apply_fused_layer_norm_matmul_tensorcore_gelu(*config)


def fused_matmul_tensorcore_add_tune(tunning=False):
    kernel_configs = [
        # [1,64,64,512,128, "swin_transform", "float16"],
        # [1,32,32,1024,256, "swin_transform", "float16"],
        [1, 16, 16, 2048, 512, "swin_transform", "float16"],
        # [1,8,8,4096,1024, "swin_transform", "float16"]
    ]
    for config in kernel_configs:
        if tunning:
            auto_tvm_tune_fused_matmul_tensorcore_add(*config)
        auto_tvm_apply_fused_matmul_tensorcore_add(*config)


def in_house_gemm(M, N, K, num_warm=0, num_bench=1, num_repeat=1):
    return swin_transformer_binding.bench_swin_ffn(M, N, K, num_warm, num_bench, num_repeat)


if __name__ == "__main__":
    fused_layer_normalization_tensorcore_gelu_tune(False)
    fused_matmul_tensorcore_add_tune(False)

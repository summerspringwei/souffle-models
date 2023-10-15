import swin_transformer_binding
import swin_trans_fc2
import cutlass_gemm
import torch
import numpy as np
import math


def test_swin_transformer_fused_mlp():
  batch_size, seq_length, in_features, out_features = 1, 256, 512, 2048
  np_fc1_weight = np.reshape(np.load("../../../data/swin-transformer-Matmul_1427_fc1_weight_512x2048.npy"), (in_features,out_features))
  np_fc2_weight = np.reshape(np.load("../../../data/swin-transformer-Matmul_1437_fc2_weight_2048x512.npy"), (out_features,in_features))
  src = torch.rand((batch_size * seq_length, in_features), dtype=torch.half, device="cuda").uniform_(-1, 1) / 16
  fc1_weight = torch.reshape(torch.from_numpy(np_fc1_weight), (in_features, out_features)).to("cuda").to(torch.float16)
  fc2_weight = torch.reshape(torch.from_numpy(np_fc2_weight), (out_features, in_features)).to("cuda").to(torch.float16)
  fc1_output, fc2_output = swin_transformer_binding.swin_fused_feed_mlp(src, fc1_weight, fc2_weight)
  print(fc2_output)


def test_swin_ffn():
  M, N, K = 256, 2048, 512
  src = torch.rand((M, K), dtype=torch.half, device="cuda").uniform_(-1, 1) / 16
  # np_fc1_weight = np.reshape(np.load("../../../data/swin-transformer-Matmul_1427_fc1_weight_512x2048.npy"), (K, N))
  # np_fc2_weight = np.reshape(np.load("../../../data/swin-transformer-Matmul_1437_fc2_weight_2048x512.npy"), (N, K))
  # fc1_weight = torch.reshape(torch.from_numpy(np_fc1_weight), (K, N)).to("cuda").to(torch.float16)
  # fc2_weight = torch.reshape(torch.from_numpy(np_fc2_weight), (N, K)).to("cuda").to(torch.float16)
  fc1_weight = torch.randn((K, N), dtype=torch.half, device="cuda").uniform_(-1, 1) / 16
  fc2_weight = torch.randn((N, K), dtype=torch.half, device="cuda").uniform_(-1, 1) / 16
  fc1_output = swin_transformer_binding.swin_ffn(src, fc1_weight, M, N, K)
  # print(fc1_output)
  # print("aaa")
  print(fc1_output.shape)
  fc2_output = swin_transformer_binding.swin_ffn(fc1_output, fc2_weight, M, K, N)
  print(fc2_output)
  # latency = swin_transformer_binding.bench_swin_ffn(M, K, N, 10, 3, 10000)
  # print(latency)


def test_bench_swin_transformer_fused_mlp():
  batch_size, seq_length, in_features, out_features = 1, 256, 512, 2048
  np_fc1_weight = np.reshape(np.load("../../../data/swin-transformer-Matmul_1427_fc1_weight_512x2048.npy"), (in_features,out_features))
  np_fc2_weight = np.reshape(np.load("../../../data/swin-transformer-Matmul_1437_fc2_weight_2048x512.npy"), (out_features,in_features))
  src = torch.rand((batch_size * seq_length, in_features), dtype=torch.half, device="cuda").uniform_(-1, 1) / 16
  fc1_weight = torch.reshape(torch.from_numpy(np_fc1_weight), (in_features, out_features)).to("cuda").to(torch.float16)
  fc2_weight = torch.reshape(torch.from_numpy(np_fc2_weight), (out_features, in_features)).to("cuda").to(torch.float16)
  latency = swin_transformer_binding.bench_swin_fused_feed_mlp(src, fc1_weight, fc2_weight, 1, 1, 1)
  print(latency)


def test_swin_trans_fc2():
  M, N, K = 256, 512, 2048
  src = torch.ones((1, M, K), dtype=torch.half, device="cuda") / 16
  weight = torch.ones((K, N), dtype=torch.half, device="cuda") / 16
  output = swin_trans_fc2.swin_trans_fc2(src, weight)
  print(output)
  # print(output.shape)


def test_cutlass_gemm():
  # cutlass_gemm.swin_trans_cutlass_gemm()
  # cutlass_gemm.swin_trans_fc2_splitK_m512n256k2048()
  # cutlass_gemm.swin_trans_fc1_m4096n512k128()
  # cutlass_gemm.swin_trans_fc2_m4096n128k512()
  # cutlass_gemm.swin_trans_fc1_m1024n1024k256()
  # cutlass_gemm.swin_trans_fc2_m1024n256k1024()
  # cutlass_gemm.swin_trans_fc1_m256n2048k512()
  # cutlass_gemm.swin_trans_fc2_m512n256k2048()
  # cutlass_gemm.swin_trans_fc1_m64n4096k1024()
  cutlass_gemm.swin_trans_fc2_m64n1024k4096()
  cutlass_gemm.swin_trans_fc2_slicedK_m64n1024k4096()
  cutlass_gemm.swin_trans_patch_merge_slicedK_m64n1024k2048()
  cutlass_gemm.swin_trans_patch_merge_slicedK_m256n512k1024()
  cutlass_gemm.swin_trans_patch_merge_slicedK_m1024n256k512()
  # a = torch.ones((256, 2048), dtype=torch.half, device="cuda") / 16
  # b = torch.ones((2048, 512), dtype=torch.half, device="cuda") / 16
  # c = cutlass_gemm.swin_trans_torch_cutlass_gemm(a, b)
  # print(c)
  # print(c.shape)

if __name__=="__main__":
  # test_swin_transformer_fused_mlp()
  # test_bench_swin_transformer_fused_mlp()
  # test_swin_ffn()
  # test_swin_trans_fc2()
  test_cutlass_gemm()
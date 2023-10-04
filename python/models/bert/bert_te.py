import tvm
from tvm import te, tir, auto_scheduler, topi, autotvm

B = 1 # batch size
S = 128 # Max sequence length
N = 12 # num heads
H = 64 # hidden size


input = te.placeholder((B, S, N*H), name="input")
attn_qkv_weight = te.placeholder((N*H, 3*N*H), name="weight")
attn_fc_weight = te.placeholder((N*H, 3*N*H), name="weight")
rk = te.reduce_axis((0, N*H))

# QKV matmul
QKV = te.compute((B, S, 3*N*H), 
  lambda b, s, h: te.sum(
    input[b, s, rk] * attn_qkv_weight[rk, h], 
    axis=[rk]))

# Split
Q = te.compute((B, S, N*H),
  lambda b, s, h: QKV[b, s, h])
K = te.compute((B, S, N*H),
  lambda b, s, h: QKV[b, s, N*H+h])
V = te.compute((B, S, N*H),
  lambda b, s, h: QKV[b, s, 2*N*H+h])
# Transpose
QT = te.compute((B*N, S, H),
  lambda n, s, h: Q[n/N, s, (n%N)*H+h])
KT = te.compute((B*N, S, H),
  lambda n, s, h: K[n/N, s, (n%N)*H+h])
VT = te.compute((B*N, H, S),
  lambda n, h, s: V[n/N, s, (n%N)*H+h])


# QK bmm
QK_output = te.compute((B, S, S), 
  lambda b, sq, sk: te.sum(
    QT[b, sq, rk] * KT[b, sk, rk], 
    axis=[rk]))
# QK mul
C = te.placeholder((1,), name="scale")
QK_output = te.compute((B, S, S),
  lambda b, sq, sk: QK_output[b, sq, sk] * C[0])
# QK Softmax
rs = te.reduce_axis((0, S), name="rs")
softmax_sum = te.compute((B, S), 
  lambda b, s: te.sum(
    tir.exp(QK_output[b, s, rs]), axis=[rs]))
softmax_norm = te.compute((B, S, S),
  lambda b, sq, sk: tir.exp(
    QK_output[b, sq, sk]) * softmax_sum[b, sq])
# Attn bmm
rs = te.reduce_axis((0, S), name='rs')
attn = te.compute((B*N, S, H), 
  lambda n, s, h: te.sum(
    softmax_norm[n, s, rs] * VT[n, h, rs], axis=[rs]))
# Transpose
attn_T = te.compute((B, S, N*H), 
  lambda b, s, h: attn[b*N+(h/H), s, h%H])
# attn FC
attn_output = te.compute((B, S, N*H), 
  lambda b, s, n: 
  te.sum(attn_T[b, s, rk] * attn_fc_weight[n, rk], axis=[rk]))
attn_output = te.compute((B, S, N*H), lambda b, s, n: input[b, s, rk] + attn_output[n, rk])

# Layernorm sum
layer_norm_sum = te.compute((B, S), 
  lambda b, s: te.sum((attn_output[b, s, rk]), axis=[rk]))
# Layernorm std
layer_norm_std = te.compute((B, S), 
  lambda b, s: te.sum(tir.power(attn_output[b, s, rk]- (layer_norm_sum[b, s]/(N*H)), 2), axis=[rk]))
# Layernorm normalize
layer_norm = te.compute((B, S, N*H), 
  lambda b, s, h: 
  (attn_output[b, s, h]-(layer_norm_sum[b, s]/(N*H))) / 
  tir.sqrt(layer_norm_std[b, s])
)

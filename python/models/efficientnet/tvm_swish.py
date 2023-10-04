import tvm

from tvm import te, tir
from tvm.script import tir as T

# 672, 14, 14, 672, 112

N, C, H, W, OC = 1, 672, 14, 14, 112

@tvm.script.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer((N, C, H, W), "float32"), Avg: T.Buffer((N, C), "float32"), 
             Norm: T.Buffer((N, C), "float32"), weight: T.Buffer((C, OC), "float32"), output: T.Buffer((N, OC), "float32")):
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # with T.block("root"):
        for n, c in T.grid(N, C):
            with T.block("init1"):
                vn, vc = T.axis.remap("SS", [n, c])
                Avg[vn, vc] = T.float32(0)
            for h, w in T.grid(H, W):
                with T.block("update1"):  
                    vn, vc, vh, vw = T.axis.remap("SSRR", [n, c, h, w])
                    Avg[vn, vc] += A[vn, vc, vh, vw]
            with T.block("norm"):
                vn, vc = T.axis.remap("SS", [n, c])
                Norm[vn, vc] = Avg[vn, vc] / T.float32(H*W)
        for n, oc in T.grid(N, OC):
              with T.block("init2"):
                  vn, voc = T.axis.remap("SS", [n, oc])
                  output[vn, voc] = T.float32(0)
              for c in range(C):
                  with T.block("update2"):
                    vn, voc, vc = T.axis.remap("SSR", [n, oc, c])
                    output[vn, voc] += (Norm[vn, vc]*weight[vc, voc])


ir_module = Module
sch = tvm.tir.Schedule(ir_module)

print(sch.mod)
mod = tvm.build(ir_module, target="llvm")  # The module for CPU backends.
print(type(mod))
# block_avg = sch.get_block("update1")
# n, c, h, w = sch.get_loops(block_avg)
# sch.bind(n, "blockIdx.x")
# sch.bind(c, "threadIdx.x")
# block_norm = sch.get_block('norm')
# sch.compute_inline(block_norm)


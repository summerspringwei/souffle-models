import tvm
from tvm import te, autotvm
from tvm.topi.cuda import tag
from tvm.topi.utils import traverse_inline, get_const_tuple
from tvm.topi.cuda.tensor_intrin import (
    intrin_wmma_load_matrix_A,
    intrin_wmma_load_matrix_W,
    intrin_wmma_store_matrix,
    intrin_wmma_gemm,
)


def nearest_power_2(num):
    for i in range(20):
        if 2**i > num:
            return 2**i


def _inline_precedence(tensor, s):
    """inline two level computation"""
    if isinstance(tensor.op, tvm.te.ComputeOp) and tag.is_injective(
            tensor.op.tag):
        s[tensor].compute_inline()
        print("inline {}".format(tensor))
    if len(tensor.op.input_tensors) > 0:
        x_normalized = tensor.op.input_tensors[0]
        if isinstance(x_normalized.op, tvm.te.ComputeOp) and tag.is_injective(
                x_normalized.op.tag):
            s[x_normalized].compute_inline()
            print("inline {}".format(x_normalized))


def _schedule_fused_precedence_dense_tensorcore(cfg, s, C):
    """Schedule dense operator using Tensorcore"""
    A, B = s[C].op.input_tensors
    if len(B.op.input_tensors) == 1 and B.op.input_tensors[0] == A:
        s[B].compute_inline()

    _inline_precedence(A, s)
    _inline_precedence(B, s)

    batch, out_dim = get_const_tuple(C.shape)
    data_dtype = A.dtype
    out_dtype = C.dtype

    # Explicit memory access
    AS = s.cache_read(A, "shared", [C])
    BS = s.cache_read(B, "shared", [C])
    AF = s.cache_read(AS, "wmma.matrix_a", [C])
    BF = s.cache_read(BS, "wmma.matrix_b", [C])
    CF = s.cache_write(C, "wmma.accumulator")
    CS = s.cache_read(CF, "shared", [C])

    # fallback support
    target = tvm.target.Target.current()
    if cfg.is_fallback:
        ref_log = autotvm.tophub.load_reference_log(target.kind.name,
                                                    target.model,
                                                    "dense_tensorcore.cuda")
        cfg.fallback_with_reference_log(ref_log)

    # Deal with op fusion, such as bias and relu
    if C.op not in s.outputs:
        s[C].compute_inline()
        C = s.outputs[0].output(0)

    # create tuning space
    cfg.define_knob("block_row_warps", [1, 2, 4])
    cfg.define_knob("block_col_warps", [1, 2, 4])
    cfg.define_knob("warp_row_tiles", [1, 2, 4])
    cfg.define_knob("warp_col_tiles", [1, 2, 4])
    cfg.define_knob("chunk", [1, 2, 4, 8])
    cfg.define_knob("offset", [0, 8])
    cfg.define_knob("offsetCS", [0, 8])
    cfg.define_knob("vec", [1, 2, 4, 8])

    if data_dtype in ["float16", "int8", "uint8"]:
        # Ensure that the default parameters are applicable when autotvm is not in use
        if batch % 32 == 0 and out_dim % 8 == 0:
            cfg.define_knob("wmma_m", [32, 16, 8])
        elif batch % 16 == 0 and out_dim % 16 == 0:
            cfg.define_knob("wmma_m", [16, 8, 32])
        elif batch % 8 == 0 and out_dim % 32 == 0:
            cfg.define_knob("wmma_m", [8, 16, 32])
        wmma_k = 16
        wmma_m = cfg["wmma_m"].val
        if wmma_m == 16:
            wmma_n = 16
        elif wmma_m == 8:
            wmma_n = 32
        elif wmma_m == 32:
            wmma_n = 8
    elif data_dtype in ["int4", "uint4"]:
        wmma_m = wmma_n = 8
        wmma_k = 32
    else:
        raise ValueError("data dtype %s is not yet supported" % data_dtype)

    warp_size = 32
    block_row_warps = cfg["block_row_warps"].val
    block_col_warps = 4 // block_row_warps
    # block_col_warps = cfg["block_col_warps"].val
    warp_row_tiles = cfg["warp_row_tiles"].val
    warp_col_tiles = cfg["warp_col_tiles"].val
    chunk = cfg["chunk"].val
    offset = cfg["offset"].val
    offsetCS = cfg["offsetCS"].val
    vec = cfg["vec"].val

    # Define the stride of intrin functions
    AS_align = chunk * wmma_k + offset
    BS_align = chunk * wmma_k + offset
    CS_align = warp_col_tiles * block_col_warps * wmma_n + offsetCS
    AS_stride = [AS_align, 1]
    BS_stride = [BS_align, 1]
    AF_stride = [wmma_k, 1]
    BF_stride = [wmma_k, 1]
    CF_stride = [warp_col_tiles * wmma_n, 1]
    CS_stride = [CS_align, 1]

    block_x = te.thread_axis("blockIdx.x")
    block_y = te.thread_axis("blockIdx.y")
    thread_x = te.thread_axis("threadIdx.x")
    thread_y = te.thread_axis("threadIdx.y")
    thread_z = te.thread_axis("threadIdx.z")

    # Schedule for dense computation
    block_factor_b = wmma_m * warp_row_tiles * block_row_warps
    block_factor_o = wmma_n * warp_col_tiles * block_col_warps
    b, o = C.op.axis
    block_i, bc = s[C].split(b, factor=block_factor_b)

    block_j, oc = s[C].split(o, factor=block_factor_o)
    s[C].reorder(block_i, block_j, bc, oc)
    t = s[C].fuse(bc, oc)
    t, vi = s[C].split(t, factor=vec)
    t, tx = s[C].split(t, factor=warp_size)
    t, ty = s[C].split(t, factor=block_row_warps)
    t, tz = s[C].split(t, factor=block_col_warps)

    s[C].bind(block_i, block_x)
    # s[x_normalized].bind(block_xb, block_x) # m
    s[C].bind(block_j, block_y)
    s[C].bind(tz, thread_z)
    s[C].bind(ty, thread_y)
    s[C].bind(tx, thread_x)
    s[C].vectorize(vi)

    # Schedule for wmma store
    s[CS].compute_at(s[C], block_j)
    bb, oo = CS.op.axis
    s[CS].storage_align(bb, CS_align - 1, CS_align)
    bb, bbi = s[CS].split(bb, factor=wmma_m)
    oo, ooi = s[CS].split(oo, factor=wmma_n)
    bb, bbii = s[CS].split(bb, factor=warp_row_tiles)
    oo, ooii = s[CS].split(oo, factor=warp_col_tiles)
    s[CS].reorder(bb, oo, bbii, ooii, bbi, ooi)
    s[CS].bind(bb, thread_y)
    s[CS].bind(oo, thread_z)

    # Schedule for wmma computation
    s[CF].compute_at(s[CS], oo)
    warp_i, warp_j = CF.op.axis
    warp_i, _ii = s[CF].split(warp_i, factor=wmma_m)
    warp_j, _jj = s[CF].split(warp_j, factor=wmma_n)
    (k, ) = CF.op.reduce_axis
    k, _k = s[CF].split(k, factor=wmma_k)
    ko, ki = s[CF].split(k, factor=chunk)
    s[CF].reorder(ko, ki, warp_i, warp_j, _ii, _jj, _k)

    # Schedule for  wmma_matrix_a load
    s[AF].compute_at(s[CF], ki)
    b, i = AF.op.axis
    b, b_ii = s[AF].split(b, factor=wmma_m)
    i, i_jj = s[AF].split(i, factor=wmma_k)
    s[AF].reorder(b, i, b_ii, i_jj)

    # Schedule for  wmma_matrix_b load
    s[BF].compute_at(s[CF], ki)
    o, i = BF.op.axis
    o, o_ii = s[BF].split(o, factor=wmma_n)
    i, i_ii = s[BF].split(i, factor=wmma_k)
    s[BF].reorder(o, i, o_ii, i_ii)

    # Schedule for A's(B's) shared memory load
    def shared_shedule(stage, strides):
        s[stage].compute_at(s[CF], ko)
        xo, yo = stage.op.axis
        s[stage].storage_align(xo, strides - 1, strides)
        t = s[stage].fuse(xo, yo)
        t, vi = s[stage].split(t, factor=vec)
        t, tx = s[stage].split(t, factor=warp_size)
        t, ty = s[stage].split(t, factor=block_row_warps)
        _, tz = s[stage].split(t, factor=block_col_warps)
        s[stage].bind(ty, thread_y)
        s[stage].bind(tz, thread_z)
        s[stage].bind(tx, thread_x)
        s[stage].vectorize(vi)

    shared_shedule(AS, AS_align)
    shared_shedule(BS, BS_align)

    shape = (wmma_m, wmma_n, wmma_k)
    AL_gemm = te.placeholder((wmma_m, wmma_k),
                             name="AL_gemm",
                             dtype=data_dtype)
    BL_gemm = te.placeholder((wmma_n, wmma_k),
                             name="BL_gemm",
                             dtype=data_dtype)
    k_gemm = te.reduce_axis((0, wmma_k), name="k_gemm")
    CL_compute = te.compute(
        (wmma_m, wmma_n),
        lambda ii, jj: te.sum(
            AL_gemm[ii, k_gemm].astype(out_dtype) * BL_gemm[jj, k_gemm].astype(
                out_dtype),
            axis=k_gemm,
        ),
        name="CL_compute",
    )

    # lower the computation loops down to TensorCore hardware intrinsics
    # by mapping the dense tensorcore to tensor intrinsics
    s[AF].tensorize(
        b_ii,
        intrin_wmma_load_matrix_A(AF_stride, AS_stride, shape, "row_major",
                                  (wmma_m, wmma_k), (wmma_m, wmma_k),
                                  data_dtype),
    )
    s[BF].tensorize(
        o_ii,
        intrin_wmma_load_matrix_W(BF_stride, BS_stride, shape, "col_major",
                                  (wmma_n, wmma_k), (wmma_n, wmma_k),
                                  data_dtype),
    )
    s[CF].tensorize(
        _ii,
        intrin_wmma_gemm(AL_gemm, BL_gemm, CL_compute, AF_stride, BF_stride,
                         CF_stride, shape))
    s[CS].tensorize(
        bbi,
        intrin_wmma_store_matrix(CS_stride, CF_stride, shape, out_dtype,
                                 (wmma_m, wmma_n), (wmma_m, wmma_n)),
    )


def _schedule_inline_precedence_batch_matmul_tensorcore(cfg, s, C):
    A, B = s[C].op.input_tensors
    if len(B.op.input_tensors) == 1 and B.op.input_tensors[0] == A:
        s[B].compute_inline()

    _inline_precedence(A, s)
    _inline_precedence(B, s)

    batch, m_dim, k_dim = get_const_tuple(A.shape)
    batch, n_dim, k_dim = get_const_tuple(B.shape)
    data_dtype = A.dtype
    out_dtype = C.dtype

    # Explicit memory access
    AS = s.cache_read(A, "shared", [C])
    BS = s.cache_read(B, "shared", [C])
    AF = s.cache_read(AS, "wmma.matrix_a", [C])
    BF = s.cache_read(BS, "wmma.matrix_b", [C])
    CF = s.cache_write(C, "wmma.accumulator")
    CS = s.cache_read(CF, "shared", [C])

    # fallback support
    target = tvm.target.Target.current()
    if cfg.is_fallback:
        ref_log = autotvm.tophub.load_reference_log(
            target.kind.name, target.model, "batch_matmul_tensorcore.cuda")
        cfg.fallback_with_reference_log(ref_log)

    # Deal with op fusion, such as bias/relu and slice after padding
    if C.op not in s.outputs and "injective" in s.outputs[0].tag:
        s[C].compute_inline()
        C = s.outputs[0].output(0)

    # create tuning space
    cfg.define_knob("block_row_warps", [1, 2, 4])
    cfg.define_knob("block_col_warps", [1, 2, 4])
    cfg.define_knob("warp_row_tiles", [1, 2, 4])
    cfg.define_knob("warp_col_tiles", [1, 2, 4])
    cfg.define_knob("chunk", [1, 2, 4, 8])
    cfg.define_knob("offset", [0, 8])
    cfg.define_knob("offsetCS", [0, 8])
    cfg.define_knob("vec", [1, 2, 4, 8])

    # Ensure that the default parameters are applicable when autotvm is not in use
    if data_dtype in ["float16", "uint8", "int8"]:
        if m_dim % 32 == 0 and n_dim % 8 == 0:
            cfg.define_knob("wmma_m", [32, 16, 8])
        elif m_dim % 16 == 0 and n_dim % 16 == 0:
            cfg.define_knob("wmma_m", [16, 8, 32])
        elif m_dim % 8 == 0 and n_dim % 32 == 0:
            cfg.define_knob("wmma_m", [8, 16, 32])
        wmma_k = 16
        wmma_m = cfg["wmma_m"].val
        if wmma_m == 16:
            wmma_n = 16
        elif wmma_m == 8:
            wmma_n = 32
        elif wmma_m == 32:
            wmma_n = 8
    elif data_dtype in ["int4", "uint4"]:
        wmma_m = wmma_n = 8
        wmma_k = 32
    else:
        raise ValueError("data dtype %s is not yet supported" % data_dtype)

    warp_size = 32
    block_row_warps = cfg["block_row_warps"].val
    block_col_warps = cfg["block_col_warps"].val
    warp_row_tiles = cfg["warp_row_tiles"].val
    warp_col_tiles = cfg["warp_col_tiles"].val
    chunk = cfg["chunk"].val
    offset = cfg["offset"].val
    offsetCS = cfg["offsetCS"].val
    vec = cfg["vec"].val

    # Define the stride of intrin functions
    AS_align = chunk * wmma_k + offset
    BS_align = chunk * wmma_k + offset
    CS_align = warp_col_tiles * block_col_warps * wmma_n + offsetCS
    AS_stride = [AS_align, 1]
    BS_stride = [BS_align, 1]
    AF_stride = [wmma_k, 1]
    BF_stride = [wmma_k, 1]
    CF_stride = [warp_col_tiles * wmma_n, 1]
    CS_stride = [CS_align, 1]

    block_x = te.thread_axis("blockIdx.x")
    block_y = te.thread_axis("blockIdx.y")
    block_z = te.thread_axis("blockIdx.z")
    thread_x = te.thread_axis("threadIdx.x")
    thread_y = te.thread_axis("threadIdx.y")
    thread_z = te.thread_axis("threadIdx.z")

    # Schedule for dense computation
    block_factor_m = wmma_m * warp_row_tiles * block_row_warps
    block_factor_n = wmma_n * warp_col_tiles * block_col_warps
    b, m, n = C.op.axis
    block_i, bc = s[C].split(m, factor=block_factor_m)
    block_j, oc = s[C].split(n, factor=block_factor_n)
    s[C].reorder(b, block_i, block_j, bc, oc)
    t = s[C].fuse(bc, oc)
    t, vi = s[C].split(t, factor=vec)
    t, tx = s[C].split(t, factor=warp_size)
    t, ty = s[C].split(t, factor=block_row_warps)
    t, tz = s[C].split(t, factor=block_col_warps)
    s[C].bind(block_i, block_x)
    s[C].bind(block_j, block_y)
    s[C].bind(b, block_z)
    s[C].bind(tz, thread_z)
    s[C].bind(ty, thread_y)
    s[C].bind(tx, thread_x)
    s[C].vectorize(vi)

    # Schedule for wmma store
    s[CS].compute_at(s[C], block_j)
    bs, bb, oo = CS.op.axis
    s[CS].storage_align(bb, CS_align - 1, CS_align)
    bb, bbi = s[CS].split(bb, factor=wmma_m)
    oo, ooi = s[CS].split(oo, factor=wmma_n)
    bb, bbii = s[CS].split(bb, factor=warp_row_tiles)
    oo, ooii = s[CS].split(oo, factor=warp_col_tiles)
    s[CS].reorder(bs, bb, oo, bbii, ooii, bbi, ooi)

    # Schedule for wmma computation
    s[CF].compute_at(s[CS], oo)
    bs, warp_i, warp_j = CF.op.axis
    warp_i, _ii = s[CF].split(warp_i, factor=wmma_m)
    warp_j, _jj = s[CF].split(warp_j, factor=wmma_n)
    (k, ) = CF.op.reduce_axis
    k, _k = s[CF].split(k, factor=wmma_k)
    ko, ki = s[CF].split(k, factor=chunk)
    s[CF].reorder(bs, ko, ki, warp_i, warp_j, _ii, _jj, _k)

    # Schedule for  wmma_matrix_a load
    s[AF].compute_at(s[CF], ki)
    bs, b, i = AF.op.axis
    b, b_ii = s[AF].split(b, factor=wmma_m)
    i, i_jj = s[AF].split(i, factor=wmma_k)
    s[AF].reorder(bs, b, i, b_ii, i_jj)

    # Schedule for  wmma_matrix_b load
    s[BF].compute_at(s[CF], ki)
    bs, o, i = BF.op.axis
    o, o_ii = s[BF].split(o, factor=wmma_n)
    i, i_ii = s[BF].split(i, factor=wmma_k)
    s[BF].reorder(bs, o, i, o_ii, i_ii)

    # Schedule for A's(B's) shared memory load
    def shared_shedule(stage, strides):
        s[stage].compute_at(s[CF], ko)
        bs, xo, yo = stage.op.axis
        s[stage].storage_align(xo, strides - 1, strides)
        t = s[stage].fuse(xo, yo)
        t, vi = s[stage].split(t, factor=vec)
        t, tx = s[stage].split(t, factor=warp_size)
        t, ty = s[stage].split(t, factor=block_row_warps)
        _, tz = s[stage].split(t, factor=block_col_warps)
        s[stage].bind(ty, thread_y)
        s[stage].bind(tz, thread_z)
        s[stage].bind(tx, thread_x)
        s[stage].vectorize(vi)

    shared_shedule(AS, AS_align)
    shared_shedule(BS, BS_align)

    shape = (wmma_m, wmma_n, wmma_k)
    AL_gemm = te.placeholder((wmma_m, wmma_k),
                             name="AL_gemm",
                             dtype=data_dtype)
    BL_gemm = te.placeholder((wmma_n, wmma_k),
                             name="BL_gemm",
                             dtype=data_dtype)
    k_gemm = te.reduce_axis((0, wmma_k), name="k_gemm")
    CL_compute = te.compute(
        (wmma_m, wmma_n),
        lambda ii, jj: te.sum(
            AL_gemm[ii, k_gemm].astype(out_dtype) * BL_gemm[jj, k_gemm].astype(
                out_dtype),
            axis=k_gemm,
        ),
        name="CL_compute",
    )

    # lower the computation loops down to TensorCore hardware intrinsics
    # by mapping the dense tensorcore to tensor intrinsics
    s[AF].tensorize(
        b_ii,
        intrin_wmma_load_matrix_A(
            AF_stride,
            AS_stride,
            shape,
            "row_major",
            (wmma_m, wmma_k),
            (wmma_m, wmma_k),
            data_dtype,
        ),
    )
    s[BF].tensorize(
        o_ii,
        intrin_wmma_load_matrix_W(
            BF_stride,
            BS_stride,
            shape,
            "col_major",
            (wmma_n, wmma_k),
            (wmma_n, wmma_k),
            data_dtype,
        ),
    )
    s[CF].tensorize(
        _ii,
        intrin_wmma_gemm(AL_gemm, BL_gemm, CL_compute, AF_stride, BF_stride,
                         CF_stride, shape),
    )
    s[CS].tensorize(
        bbi,
        intrin_wmma_store_matrix(CS_stride, CF_stride, shape, out_dtype,
                                 (wmma_m, wmma_n), (wmma_m, wmma_n)),
    )


def _schedule_n_dim_to_one_vp_batch_matmul_tensorcore(cfg, s, C):
    """Schedule the computation of n_dim to one virtual processor"""
    A, B = s[C].op.input_tensors
    if len(B.op.input_tensors) == 1 and B.op.input_tensors[0] == A:
        s[B].compute_inline()
    batch, m_dim, k_dim = get_const_tuple(A.shape)
    batch, n_dim, k_dim = get_const_tuple(B.shape)
    data_dtype = A.dtype
    out_dtype = C.dtype

    # Explicit memory access
    AS = s.cache_read(A, "shared", [C])
    BS = s.cache_read(B, "shared", [C])
    AF = s.cache_read(AS, "wmma.matrix_a", [C])
    BF = s.cache_read(BS, "wmma.matrix_b", [C])
    CF = s.cache_write(C, "wmma.accumulator")
    CS = s.cache_read(CF, "shared", [C])

    # fallback support
    target = tvm.target.Target.current()
    if cfg.is_fallback:
        ref_log = autotvm.tophub.load_reference_log(
            target.kind.name, target.model, "batch_matmul_tensorcore.cuda")
        cfg.fallback_with_reference_log(ref_log)

    # Deal with op fusion, such as bias/relu and slice after padding
    if C.op not in s.outputs and "injective" in s.outputs[0].tag:
        s[C].compute_inline()
        C = s.outputs[0].output(0)

    # create tuning space
    cfg.define_knob("block_row_warps", [1, 2, 3, 4, 6, 8, 12, 16])
    # cfg.define_knob("block_col_warps", [1, 3, 6, 12]) # Optimize for n_dim=384
    # cfg.define_knob("block_col_warps", [1, 2, 4])
    cfg.define_knob("warp_row_tiles", [1, 2, 4, 6, 8])
    cfg.define_knob("warp_col_tiles",
                    [1, 2, 4, 8, 16])  # Optimize for n_dim=384
    # cfg.define_knob("warp_col_tiles", [1, 2, 4])
    cfg.define_knob("chunk", [1, 2, 3, 4, 6, 8, 12, 16])
    cfg.define_knob("offset", [0, 8])
    cfg.define_knob("offsetCS", [0, 8])
    cfg.define_knob("vec", [1, 2, 4, 8])

    # Ensure that the default parameters are applicable when autotvm is not in use
    if data_dtype in ["float16", "uint8", "int8"]:
        if m_dim % 32 == 0 and n_dim % 8 == 0:
            cfg.define_knob("wmma_m", [32, 16, 8])
            # cfg.define_knob("wmma_m", [8])
        elif m_dim % 16 == 0 and n_dim % 16 == 0:
            cfg.define_knob("wmma_m", [16, 8, 32])
        elif m_dim % 8 == 0 and n_dim % 32 == 0:
            cfg.define_knob("wmma_m", [8, 16, 32])
        wmma_k = 16
        wmma_m = cfg["wmma_m"].val
        if wmma_m == 16:
            wmma_n = 16
        elif wmma_m == 8:
            wmma_n = 32
        elif wmma_m == 32:
            wmma_n = 8
    elif data_dtype in ["int4", "uint4"]:
        wmma_m = wmma_n = 8
        wmma_k = 32
    else:
        raise ValueError("data dtype %s is not yet supported" % data_dtype)
    # print("wmma_m: {}, wmma_n: {}, wmma_k: {}".format(wmma_m, wmma_n, wmma_k))
    warp_size = 32
    block_row_warps = cfg["block_row_warps"].val
    # block_col_warps = cfg["block_col_warps"].val
    # warp_row_tiles = m_dim // wmma_m // cfg["block_row_warps"].val
    warp_row_tiles = cfg["warp_row_tiles"].val
    # warp_col_tiles = cfg["warp_col_tiles"].val
    chunk = cfg["chunk"].val
    offset = cfg["offset"].val
    offsetCS = cfg["offsetCS"].val
    vec = cfg["vec"].val

    # Let each block compute compute only wmma_m * n_dim output elements
    # warp_row_tiles, block_row_warps = 1, 1
    # Let block_factor_n == n_dim
    warp_col_tiles, block_col_warps = cfg[
        "warp_col_tiles"].val, n_dim // wmma_n // cfg["warp_col_tiles"].val
    # print("warp_col_tiles: {}, block_col_warps: {}".format(warp_col_tiles, block_col_warps))
    # Define the stride of intrin functions
    AS_align = chunk * wmma_k + offset
    BS_align = chunk * wmma_k + offset
    CS_align = warp_col_tiles * block_col_warps * wmma_n + offsetCS
    AS_stride = [AS_align, 1]
    BS_stride = [BS_align, 1]
    AF_stride = [wmma_k, 1]
    BF_stride = [wmma_k, 1]
    CF_stride = [warp_col_tiles * wmma_n, 1]
    CS_stride = [CS_align, 1]

    block_x = te.thread_axis("blockIdx.x")
    block_y = te.thread_axis("blockIdx.y")
    block_z = te.thread_axis("blockIdx.z")
    thread_x = te.thread_axis("threadIdx.x")
    thread_y = te.thread_axis("threadIdx.y")
    thread_z = te.thread_axis("threadIdx.z")

    # Schedule for dense computation
    # 16*warp_row_tiles*block_row_warps
    block_factor_m = wmma_m * warp_row_tiles * block_row_warps
    block_factor_n = wmma_n * warp_col_tiles * block_col_warps
    if block_factor_n != n_dim:
        raise ValueError("block_factor_n is {} while n_dim is {}".format(
            block_factor_n, n_dim))

    # 1*12, 384, 384
    b, m, n = C.op.axis
    # block_i=384/block_factor_m, bc=block_factor_m
    block_i, bc = s[C].split(m, factor=block_factor_m)
    # block_j=384/block_factor_n, oc=block_factor_n
    # Let block_j==1 and oc=n_dim, we can make a block compute last dim (n_dim) output elements
    block_j, oc = s[C].split(n, factor=block_factor_n)
    # 12, 384/block_factor_m, 384/block_factor_n, block_factor_m, block_factor_n
    s[C].reorder(b, block_i, block_j, bc, oc)
    # t = block_factor_m * block_factor_n
    t = s[C].fuse(bc, oc)
    # t = block_factor_m * block_factor_n / vec, vi = vec
    t, vi = s[C].split(t, factor=vec)
    # t = block_factor_m * block_factor_n / vec / warp_size, tx=warp_size
    t, tx = s[C].split(t, factor=warp_size)
    # t = block_factor_m * block_factor_n / vec / warp_size / block_row_warps, ty = block_row_warps
    t, ty = s[C].split(t, factor=block_row_warps)
    # t = block_factor_m * block_factor_n / vec / warp_size / block_row_warps / block_col_warps, tz = block_col_warps
    t, tz = s[C].split(t, factor=block_col_warps)

    s[C].bind(block_i, block_x)  # block_x = 384 / block_factor_m
    s[C].bind(block_j, block_y)  # block_y = 384 / block_factor_n
    s[C].bind(b, block_z)  # block_z = 12
    s[C].bind(tz, thread_z)  # tz = block_col_warps
    s[C].bind(ty, thread_y)  # ty = block_row_warps
    s[C].bind(tx, thread_x)  # tx=warp_size
    s[C].vectorize(vi)

    # Schedule for wmma store
    s[CS].compute_at(s[C], block_j)
    # 12, block_factor_m, block_factor_n
    bs, bb, oo = CS.op.axis
    s[CS].storage_align(bb, CS_align - 1, CS_align)
    # bb= warp_row_tiles * block_row_warps, bbi=wmma_m
    bb, bbi = s[CS].split(bb, factor=wmma_m)
    # oo, ooi = warp_col_tiles * block_col_warps, ooi=wmma_n
    oo, ooi = s[CS].split(oo, factor=wmma_n)
    # bb=block_row_warps, bbii=warp_row_tiles
    bb, bbii = s[CS].split(bb, factor=warp_row_tiles)
    # oo=block_col_warps, ooii=warp_col_tiles
    oo, ooii = s[CS].split(oo, factor=warp_col_tiles)
    # bs=12, bb=block_row_warps, oo=block_col_warps, bbii=warp_row_tiles, ooii=warp_col_tiles, bbi=wmma_m, ooi=wmma_n
    s[CS].reorder(bs, bb, oo, bbii, ooii, bbi, ooi)

    # Schedule for wmma computation
    # bbii=warp_row_tiles, ooii=warp_col_tiles, bbi=wmma_m, ooi=wmma_n
    s[CF].compute_at(s[CS], oo)
    # bs=12, warp_i=wrap_row_tiles*wmma_m, warp_j=warp_col_tiles*wmma_n
    bs, warp_i, warp_j = CF.op.axis
    # warp_i=warp_row_tiles, _ii=wmma_m
    warp_i, _ii = s[CF].split(warp_i, factor=wmma_m)
    # warp_j=warp_col_tiles, _j=wmma_n
    warp_j, _jj = s[CF].split(warp_j, factor=wmma_n)
    # 64
    (k, ) = CF.op.reduce_axis
    # k=64/16, _k=16
    k, _k = s[CF].split(k, factor=wmma_k)
    # ko=16/chunk, ki=chunk
    ko, ki = s[CF].split(k, factor=chunk)
    # bs=12, ko=16/chunk, ki=chunk, warp_i= warp_row_tiles, warp_j=warp_col_tiles, _ii=wmm_n, _jj=wmm_n, _k=16
    s[CF].reorder(bs, ko, ki, warp_i, warp_j, _ii, _jj, _k)

    # Schedule for wmma_matrix_a load
    s[AF].compute_at(s[CF], ki)
    bs, b, i = AF.op.axis
    b, b_ii = s[AF].split(b, factor=wmma_m)
    i, i_jj = s[AF].split(i, factor=wmma_k)
    s[AF].reorder(bs, b, i, b_ii, i_jj)

    # Schedule for  wmma_matrix_b load
    s[BF].compute_at(s[CF], ki)
    bs, o, i = BF.op.axis
    o, o_ii = s[BF].split(o, factor=wmma_n)
    i, i_ii = s[BF].split(i, factor=wmma_k)
    s[BF].reorder(bs, o, i, o_ii, i_ii)

    # Schedule for A's(B's) shared memory load
    def shared_shedule(stage, strides):
        s[stage].compute_at(s[CF], ko)
        bs, xo, yo = stage.op.axis
        s[stage].storage_align(xo, strides - 1, strides)
        t = s[stage].fuse(xo, yo)
        t, vi = s[stage].split(t, factor=vec)
        t, tx = s[stage].split(t, factor=warp_size)
        t, ty = s[stage].split(t, factor=block_row_warps)
        _, tz = s[stage].split(t, factor=block_col_warps)
        s[stage].bind(ty, thread_y)
        s[stage].bind(tz, thread_z)
        s[stage].bind(tx, thread_x)
        s[stage].vectorize(vi)

    shared_shedule(AS, AS_align)
    shared_shedule(BS, BS_align)

    shape = (wmma_m, wmma_n, wmma_k)
    AL_gemm = te.placeholder((wmma_m, wmma_k),
                             name="AL_gemm",
                             dtype=data_dtype)
    BL_gemm = te.placeholder((wmma_n, wmma_k),
                             name="BL_gemm",
                             dtype=data_dtype)
    k_gemm = te.reduce_axis((0, wmma_k), name="k_gemm")
    CL_compute = te.compute(
        (wmma_m, wmma_n),
        lambda ii, jj: te.sum(
            AL_gemm[ii, k_gemm].astype(out_dtype) * BL_gemm[jj, k_gemm].astype(
                out_dtype),
            axis=k_gemm,
        ),
        name="CL_compute",
    )

    # lower the computation loops down to TensorCore hardware intrinsics
    # by mapping the dense tensorcore to tensor intrinsics
    s[AF].tensorize(
        b_ii,
        intrin_wmma_load_matrix_A(
            AF_stride,
            AS_stride,
            shape,
            "row_major",
            (wmma_m, wmma_k),
            (wmma_m, wmma_k),
            data_dtype,
        ),
    )
    s[BF].tensorize(
        o_ii,
        intrin_wmma_load_matrix_W(
            BF_stride,
            BS_stride,
            shape,
            "col_major",
            (wmma_n, wmma_k),
            (wmma_n, wmma_k),
            data_dtype,
        ),
    )
    s[CF].tensorize(
        _ii,
        intrin_wmma_gemm(AL_gemm, BL_gemm, CL_compute, AF_stride, BF_stride,
                         CF_stride, shape),
    )
    s[CS].tensorize(
        bbi,
        intrin_wmma_store_matrix(CS_stride, CF_stride, shape, out_dtype,
                                 (wmma_m, wmma_n), (wmma_m, wmma_n)),
    )

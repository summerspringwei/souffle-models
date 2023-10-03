#pragma once
#include <cuda_fp16.h>

namespace souffle {
namespace swin_trans {

enum GPUParams {
    kNumberSM = 82,
    kElementwiseBlockThreads = 32*8,
};

enum GPT2LargeParams {
    kBatchSize = 1,
    kSeqLength = 384,
    kHeadNum = 20,
    kHeadSize = 64,
    kLayerNum = 36,
    kHiddenSize = 4,
    kHiddenDim = kHeadNum * kHeadSize,
};


enum GPTGEMMParams {
    kWmmaM = 16,
    kWmmaN = 16,
    kWmmaK = 16,
    kInputSkew = 8,
    kAccSkew = 8,
    kStage = 3,
    kWarpSize = 32,
};


namespace FeedForwardFC1Params{
    enum FC1Params {
        // M=5120, N=384, K=1280
        // ctm_m=64, cta_n=128, cta_k=32
        KGEMMFFM = 5120,
        KGEMMFFN = 384,
        KGEMMFFK = 1280,

        kWarpRowTiles = 4,
        kWarpColTiles = 3,
        kChunkK = 2,
        kBlockRowWarps = 2,
        kBlockColWarps = 2,
        
        kBlockThreads = kBlockRowWarps * kBlockColWarps * kWarpSize,
        kBlockRowTiles = kBlockRowWarps * kWarpRowTiles,
        kBlockColTiles = kBlockColWarps * kWarpColTiles,
        kGridBlocks = KGEMMFFM / kBlockRowTiles / kWmmaM * KGEMMFFN / kBlockColTiles / kWmmaN,
        kSharedMemory =
        (kStage *
         (kChunkK * kWmmaK *
              (kBlockRowWarps * FeedForwardFC1Params::kBlockRowTiles * kWmmaM +
               kInputSkew) +
          kBlockColWarps * FeedForwardFC1Params::kBlockColTiles * kWmmaN *
              (kChunkK * kWmmaK + kInputSkew))) *
        sizeof(half),
    };

} // namespace FeedForwardFC1Params

namespace FeedForwardFC1LimitedBlocksParams{
    enum FC1Params {
        // M=5120, N=384, K=1280
        // ctm_m=64, cta_n=128, cta_k=32
        KGEMMFFM = 5120,
        KGEMMFFN = 384,
        KGEMMFFK = 1280,
        kGEMMFFB = 1,

        kWarpRowTiles = 4,
        kWarpColTiles = 3,
        kMTiles = 2,
        kNTiles = 1,
        kChunkK = 2,
        kBlockRowWarps = 2,
        kBlockColWarps = 2,
        
        kBlockRowTiles = kBlockRowWarps * kWarpRowTiles,
        kBlockColTiles = kBlockColWarps * kWarpColTiles,
        kBlockThreads = kBlockRowWarps * kBlockColWarps * kWarpSize,
        kGridBlocks = (KGEMMFFM / kBlockRowTiles / kWmmaM / kMTiles) * 
            (KGEMMFFN / kBlockColTiles / kWmmaN / kNTiles),
        
        kSharedMemory = (kStage *
         (kChunkK * kWmmaK *
              (kBlockRowWarps *
                   FeedForwardFC1LimitedBlocksParams::kBlockRowTiles * kWmmaM +
               kInputSkew) +
          kBlockColWarps * FeedForwardFC1LimitedBlocksParams::kBlockColTiles *
              kWmmaN * (kChunkK * kWmmaK + kInputSkew))) *
        sizeof(half),
    };
} // namespace FeedForwardFC1LimitedBlocksParams


// weight(k, n) * input(m, k) = output(m, n):
//  (5120, 1280) * (5120, 384) -> (1280, 384)
namespace FeedForwardFC2Params{
    enum FC2Params {
        KGEMMFFM = 512,
        KGEMMFFN = 256,
        KGEMMFFK = 2048,

        kGemmK6BlockRowTiles = 2,
        kGemmK6BlockColTiles = 2,
        // May set kGEEMK6BlockSliceKTiles = 5 for A100
        kGemmK6BlockSliceKTiles = 4,
        kChunkK = 4,
        kBlockRowWarps = 2,
        kBlockColWarps = 2,

        kBlockThreads = kBlockRowWarps * kBlockColWarps * kWarpSize,
        kGridBlocks = (KGEMMFFM / (kGemmK6BlockRowTiles * kWmmaM)) *
            (KGEMMFFN / (kGemmK6BlockColTiles * kWmmaN)),

        kSharedMemory =
            (kStage * (kGemmK6BlockSliceKTiles * kWmmaK *
                        (kGemmK6BlockRowTiles * kWmmaM + kInputSkew) +
                    kGemmK6BlockColTiles * kWmmaN *
                        (kGemmK6BlockSliceKTiles * kWmmaK + kInputSkew))) *
            sizeof(half),
    };
} // namespace FeedForwardFC2Params

} // namespace gpt2
} // namespace souffle

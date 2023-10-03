#pragma once

#include <cuda.h>
#include <cuda_fp16.h>

namespace fuselage::experiments::networks::bert {

enum BertScaleParams {
    kBatchSize = 1,
    kSeqLength = 384,
    kHeadNum = 12,
    kHeadSize = 64,
    kLayerNum = 12,
    kHiddenSize = 4,
    kHiddenDim = kHeadNum * kHeadSize,
};

enum BertGemmParams {
    kWmmaM = 16,
    kWmmaN = 16,
    kWmmaK = 16,
    kChunkK = 4,
    kStage = 3,
    kBlockRowWarps = 2,
    kBlockColWarps = 2,
    kWarpSize = 32,
    kBlockThreads = kBlockRowWarps * kBlockColWarps * kWarpSize,
    kInputSkew = 8,
    kAccSkew = 8,

    kGemmK1WarpRowTiles = 1,
    kGemmK1WarpColTiles = 3,
    kGemmK1BatchedNum = 1,

    kGemmK2WarpRowTiles = 4,
    kGemmK2WarpColTiles = 4,
    kGemmK2BatchedNum = kHeadNum,

    kGemmK3WarpRowTiles = 2,
    kGemmK3WarpColTiles = 2,
    kGemmK3BatchedNum = kHeadNum,

    kGemmK4WarpRowTiles = 2,
    kGemmK4WarpColTiles = 2,

    kGemmK5WarpRowTiles = 4,
    kGemmK5WarpColTiles = 3,

    kGemmK6BlockRowTiles = 4,
    kGemmK6BlockColTiles = 4,
    kGemmK6BlockSliceKTiles = 4,
};

struct BertWordVec {
    half data[kHiddenDim];
};

struct BertWeight {
    half data[kHiddenDim][kHiddenDim];
};

struct BertAttrMask {
    half data[kBatchSize][kSeqLength][kSeqLength];
};

#pragma pack(push, 1)
struct BertInput {
    BertWordVec words[kBatchSize][kSeqLength];
};
#pragma pack(pop)


} // namespace fuselage::experiments::networks::bert
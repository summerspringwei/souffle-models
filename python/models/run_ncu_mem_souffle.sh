#!/bin/bash

set -x
DIRPATH=/workspace/souffle-models/python/models

NCU_ARGS="--metrics dram__bytes_read,gpu__time_duration --clock-control none --target-processes all"

# cd ${DIRPATH}/bert
# BERT Pass
NAME="souffle_bert_O4"
if [ -n "${SOUFFLE_RUN}" ] && [ "${SOUFFLE_RUN}" = "TRUE" ]; then
  ncu ${NCU_ARGS} -o ncu-${NAME} -f --target-processes all python3 souffle_bert.py O4 1 1
  ncu -i ./ncu-${NAME}.ncu-rep --csv --page raw  | grep -v "at::native*" | grep -v "at_cuda_detail" > ncu-${NAME}.csv
fi
SOUFFLE_BERT_MEM=$(python3 ../../extract_ncu_cuda_mem_read.py ncu-${NAME}.csv)
SOUFFLE_BERT_NUM_KERNELS=$(wc -l ncu-${NAME}-dram_bytes_read.csv | awk '{ print $1 }')
bert_layer=12
SOUFFLE_BERT_MEM=$(python3 -c "print(${SOUFFLE_BERT_MEM} * ${bert_layer})")
SOUFFLE_BERT_NUM_KERNELS=$(python3 -c "print(${SOUFFLE_BERT_NUM_KERNELS} * ${bert_layer})")


# ResNext Pass
cd ${DIRPATH}/resnext
NAME="souffle_resnext_O4"
if [ -n "${SOUFFLE_RUN}" ] && [ "${SOUFFLE_RUN}" = "TRUE" ]; then
ncu ${NCU_ARGS} -o ncu-${NAME} -f --target-processes all \
  python3 run_souffle_resnext.py O2 1 1 | tee resnext_O4_ncu.txt 2>&1
ncu -i ./ncu-${NAME}.ncu-rep --csv --page raw  | grep -v "at::native*" > ncu-${NAME}.csv
fi
SOUFFLE_RESNEXT_MEM=$(python3 ../../extract_ncu_cuda_mem_read.py ncu-${NAME}.csv)
SOUFFLE_RESNEXT_NUM_KERNELS=$(wc -l ncu-${NAME}-dram_bytes_read.csv | awk '{ print $1 }')

# LSTM Pass
cd ${DIRPATH}/lstm
NAME="souffle_lstm_O4"
if [ -n "${SOUFFLE_RUN}" ] && [ "${SOUFFLE_RUN}" = "TRUE" ]; then
ncu ${NCU_ARGS} -o ncu-${NAME} -f \
 python3 souffle_lstm.py O3 1 1
ncu -i ./ncu-${NAME}.ncu-rep --csv --page raw  | grep -v "at::native*" > ncu-${NAME}.csv
fi
SOUFFLE_LSTM_MEM=$(python3 ../../extract_ncu_cuda_mem_read.py ncu-${NAME}.csv)
SOUFFLE_LSTM_NUM_KERNELS=$(wc -l ncu-${NAME}-dram_bytes_read.csv | awk '{ print $1 }')

# EfficientNet Pass
cd ${DIRPATH}/efficientnet
NAME="souffle_efficientnet_O3"
if [ -n "${SOUFFLE_RUN}" ] && [ "${SOUFFLE_RUN}" = "TRUE" ]; then
ncu ${NCU_ARGS} -o ncu-${NAME} -f python3 souffle_efficientnet.py O3 1 1
ncu -i ./ncu-${NAME}.ncu-rep --csv --page raw | grep -v "at::native" > ncu-${NAME}.csv
fi
SOUFFLE_EFFICIENTNET_MEM=$(python3 ../../extract_ncu_cuda_mem_read.py ncu-${NAME}.csv)
SOUFFLE_EFFICIENTNET_NUM_KERNELS=$(wc -l ncu-${NAME}-dram_bytes_read.csv | awk '{ print $1 }')


# SwinTrans. Pass
cd ${DIRPATH}/swin_transformer
NAME="souffle_swin_trans_O4"
if [ -n "${SOUFFLE_RUN}" ] && [ "${SOUFFLE_RUN}" = "TRUE" ]; then
ncu ${NCU_ARGS} -o ncu-${NAME} -f python3 souffle_swin_trans.py O4 1 1
ncu -i ./ncu-${NAME}.ncu-rep --csv --page raw  | grep -v "at::native" > ncu-${NAME}.csv
fi
SOUFFLE_SWIN_TRANS_MEM=$(python3 ../../extract_ncu_cuda_mem_read.py ncu-${NAME}.csv)
SOUFFLE_SWIN_TRANS_NUM_KERNELS=$(wc -l ncu-${NAME}-dram_bytes_read.csv | awk '{ print $1 }')



# MMOE
cd ${DIRPATH}/mmoe
NAME="souffle_mmoe_O3"
if [ -n "${SOUFFLE_RUN}" ] && [ "${SOUFFLE_RUN}" = "TRUE" ]; then
ncu ${NCU_ARGS} -o ncu-${NAME} -f python3 souffle_mmoe.py O3 1 1
ncu -i ./ncu-${NAME}.ncu-rep --csv --page raw  | grep -v "at::native*" > ncu-${NAME}.csv
fi
SOUFFLE_MMoE_MEM=$(python3 ../../extract_ncu_cuda_mem_read.py ncu-${NAME}.csv)
SOUFFLE_MMoE_NUM_KERNELS=$(wc -l ncu-${NAME}-dram_bytes_read.csv | awk '{ print $1 }')
cd ${DIRPATH}


echo "Souffle: ," ${SOUFFLE_BERT_MEM}, ${SOUFFLE_RESNEXT_MEM}, \
  ${SOUFFLE_LSTM_MEM}, ${SOUFFLE_EFFICIENTNET_MEM}, \
  ${SOUFFLE_SWIN_TRANS_MEM}, ${SOUFFLE_MMoE_MEM} > tee table5_souffle.csv
echo "Souffle: ," ${SOUFFLE_BERT_NUM_KERNELS}, ${SOUFFLE_RESNEXT_NUM_KERNELS}, \
  ${SOUFFLE_LSTM_NUM_KERNELS}, ${SOUFFLE_EFFICIENTNET_NUM_KERNELS}, \
  ${SOUFFLE_SWIN_TRANS_NUM_KERNELS}, ${SOUFFLE_MMoE_NUM_KERNELS} >> tee table5_souffle.csv

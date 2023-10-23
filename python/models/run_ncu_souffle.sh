#!/bin/bash

set -x
DIRPATH=/workspace/souffle-models/python/models

NCU_ARGS="--metrics dram__bytes_read,gpu__time_duration --clock-control none --target-processes all"

cd ${DIRPATH}/bert
# BERT Pass
if [ -n "${SOUFFLE_RUN}" ] && [ "${SOUFFLE_RUN}" = "TRUE" ]; then
  ncu ${NCU_ARGS} -o ncu-souffle_bert_O4 -f  python3 souffle_bert.py O4 1 1
  ncu -i ./ncu-souffle_bert_O4.ncu-rep --csv --page raw  | grep -v "at::native*" | grep -v "at_cuda_detail" > ncu-souffle_bert_O4.csv
fi
SOUFFLE_BERT_LATENCY=$(python3 ../../extract_ncu_cuda_kernel_latency.py ncu-souffle_bert_O4.csv)
bert_layer=12
SOUFFLE_BERT_LATENCY=$(python3 -c "print(${SOUFFLE_BERT_LATENCY} * ${bert_layer})")

# ResNext Pass
cd ${DIRPATH}/resnext
if [ -n "${SOUFFLE_RUN}" ] && [ "${SOUFFLE_RUN}" = "TRUE" ]; then
ncu ${NCU_ARGS} -o ncu-souffle_resnext_O4 -f  \
  python3 run_souffle_resnext.py O2 1 1 | tee resnext_O4_ncu.txt 2>&1
ncu -i ./ncu-souffle_resnext_O4.ncu-rep --csv --page raw  | grep -v "at::native*" > ncu-souffle_resnext_O4.csv
fi
SOUFFLE_RESNEXT_LATENCY=$(python3 ../../extract_ncu_cuda_kernel_latency.py ncu-souffle_resnext_O4.csv)

# LSTM Pass
cd ${DIRPATH}/lstm
if [ -n "${SOUFFLE_RUN}" ] && [ "${SOUFFLE_RUN}" = "TRUE" ]; then
ncu ${NCU_ARGS} -o ncu-souffle_lstm_O4 -f  \
 python3 souffle_lstm.py O3 1 1
ncu -i ./ncu-souffle_lstm_O4.ncu-rep --csv --page raw  | grep -v "at::native*" > ncu-souffle_lstm_O4.csv
fi
SOUFFLE_LSTM_LATENCY=$(python3 ../../extract_ncu_cuda_kernel_latency.py ncu-souffle_lstm_O4.csv)

# EfficientNet Pass
cd ${DIRPATH}/efficientnet
if [ -n "${SOUFFLE_RUN}" ] && [ "${SOUFFLE_RUN}" = "TRUE" ]; then
ncu ${NCU_ARGS} -o ncu-souffle_efficientnet_O3 -f  python3 souffle_efficientnet.py O3 1 1
ncu -i ./ncu-souffle_efficientnet_O3.ncu-rep --csv --page raw | grep -v "at::native" > ncu-souffle_efficientnet_O4.csv
fi
SOUFFLE_EFFICIENTNET_LATENCY=$(python3 ../../extract_ncu_cuda_kernel_latency.py ncu-souffle_efficientnet_O4.csv)

# SwinTrans. Pass
cd ${DIRPATH}/swin_transformer
if [ -n "${SOUFFLE_RUN}" ] && [ "${SOUFFLE_RUN}" = "TRUE" ]; then
ncu ${NCU_ARGS} -o ncu-souffle_swin_trans_O4 -f  python3 souffle_swin_trans.py O4 1 1
ncu -i ./ncu-souffle_swin_trans_O4.ncu-rep --csv --page raw  | grep -v "at::native" > ncu-souffle_swin_trans_O4.csv
fi
SOUFFLE_SWIN_TRANS_LATENCY=$(python3 ../../extract_ncu_cuda_kernel_latency.py ncu-souffle_swin_trans_O4.csv)

# MMOE
cd ${DIRPATH}/mmoe
if [ -n "${SOUFFLE_RUN}" ] && [ "${SOUFFLE_RUN}" = "TRUE" ]; then
ncu ${NCU_ARGS} -o ncu-souffle_mmoe_O2 -f  python3 souffle_mmoe.py O2 1 1
ncu -i ./ncu-souffle_mmoe_O2.ncu-rep --csv --page raw  | grep -v "at::native*" > ncu-souffle_mmoe_O2.csv
fi
SOUFFLE_MMoE_LATENCY=$(python3 ../../extract_ncu_cuda_kernel_latency.py ncu-souffle_mmoe_O2.csv)
cd ${DIRPATH}

echo "Souffle: ," ${SOUFFLE_BERT_LATENCY}, ${SOUFFLE_RESNEXT_LATENCY}, \
  ${SOUFFLE_LSTM_LATENCY}, ${SOUFFLE_EFFICIENTNET_LATENCY}, \
  ${SOUFFLE_SWIN_TRANS_LATENCY}, ${SOUFFLE_MMoE_LATENCY} | tee table3_souffle.csv

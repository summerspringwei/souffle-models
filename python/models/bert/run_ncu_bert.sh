#!/bin/bash
set -x

bert_layers=12

NCU_ARGS="--metrics dram__bytes_read,gpu__time_duration --clock-control none --target-processes all"

if [ -n "${SOUFFLE_RUN}" ] && [ "${SOUFFLE_RUN}" = "TRUE" ]; then
ncu ${NCU_ARGS} -o ncu-souffle_bert_O0 -f  \
   python3 souffle_bert.py O0 1 1
fi
ncu -i ./ncu-souffle_bert_O0.ncu-rep --csv --page raw > ncu-souffle_bert_O0.csv
BERT_O0_LATENCY=$(python3 ../../extract_ncu_cuda_kernel_latency.py ncu-souffle_bert_O0.csv)

# Filter out torch kernels
if [ -n "${SOUFFLE_RUN}" ] && [ "${SOUFFLE_RUN}" = "TRUE" ]; then
ncu ${NCU_ARGS} -o ncu-souffle_bert_O1 -f  \
   python3 souffle_bert.py O1 1 1
fi
ncu -i ./ncu-souffle_bert_O1.ncu-rep --csv --page raw | grep -v "at::native*" | grep -v "void at_cuda_detail*" > ncu-souffle_bert_O1.csv
BERT_O1_LATENCY=$(python3 ../../extract_ncu_cuda_kernel_latency.py ncu-souffle_bert_O1.csv)
BERT_O1_LATENCY=$(python3 -c "print(${BERT_O1_LATENCY} * ${bert_layers})")

if [ -n "${SOUFFLE_RUN}" ] && [ "${SOUFFLE_RUN}" = "TRUE" ]; then
ncu ${NCU_ARGS} -o ncu-souffle_bert_O2 -f  \
   python3 souffle_bert.py O2 1 1
fi
ncu -i ./ncu-souffle_bert_O2.ncu-rep --csv --page raw | grep -v "at::native*" | grep -v "void at_cuda_detail*" > ncu-souffle_bert_O2.csv
BERT_O2_LATENCY=$(python3 ../../extract_ncu_cuda_kernel_latency.py ncu-souffle_bert_O2.csv)
BERT_O2_LATENCY=$(python3 -c "print(${BERT_O2_LATENCY} * ${bert_layers})")

if [ -n "${SOUFFLE_RUN}" ] && [ "${SOUFFLE_RUN}" = "TRUE" ]; then
ncu ${NCU_ARGS}  -o ncu-souffle_bert_O3 -f  \
 python3 souffle_bert.py O3 1 1
fi
ncu -i ./ncu-souffle_bert_O3.ncu-rep --csv --page raw | grep -v "at::native*" | grep -v "void at_cuda_detail*" > ncu-souffle_bert_O3.csv
BERT_O3_LATENCY=$(python3 ../../extract_ncu_cuda_kernel_latency.py ncu-souffle_bert_O3.csv)
BERT_O3_LATENCY=$(python3 -c "print(${BERT_O3_LATENCY} * ${bert_layers})")

if [ -n "${SOUFFLE_RUN}" ] && [ "${SOUFFLE_RUN}" = "TRUE" ]; then
ncu ${NCU_ARGS} -o ncu-souffle_bert_O4 -f  \
  python3 souffle_bert.py O4 1 1
fi
ncu -i ./ncu-souffle_bert_O4.ncu-rep --csv --page raw  | grep -v "at::native*" | grep -v "void at_cuda_detail*" > ncu-souffle_bert_O4.csv
BERT_O4_LATENCY=$(python3 ../../extract_ncu_cuda_kernel_latency.py ncu-souffle_bert_O4.csv)
BERT_O4_LATENCY=$(python3 -c "print(${BERT_O4_LATENCY} * ${bert_layers})")


# echo "BERT:", ${BERT_O0_LATENCY}, ${BERT_O1_LATENCY}, \
#   ${BERT_O2_LATENCY}, ${BERT_O3_LATENCY}, ${BERT_O4_LATENCY} | tee table4_bert.csv

python3 -c "print('BERT:, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}'.format(${BERT_O0_LATENCY}, ${BERT_O1_LATENCY}, ${BERT_O2_LATENCY}, ${BERT_O3_LATENCY}, ${BERT_O4_LATENCY}))" | tee table4_bert.csv

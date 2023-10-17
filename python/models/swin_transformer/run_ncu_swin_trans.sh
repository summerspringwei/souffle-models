#!/bin/bash
set -x

NCU_ARGS="--metrics dram__bytes_read,gpu__time_duration --clock-control none --target-processes all"

if [ -n "${SOUFFLE_RUN}" ] && [ "${SOUFFLE_RUN}" = "TRUE" ]; then
ncu ${NCU_ARGS} -o ncu-souffle_swin_trans_O0 -f \
  python3 souffle_swin_trans.py O0 1 1
fi
ncu -i ./ncu-souffle_swin_trans_O0.ncu-rep --csv --page raw > ncu-souffle_swin_trans_O0.csv
O0_LATENCY=$(python3 ../../extract_ncu_cuda_kernel_latency.py ncu-souffle_swin_trans_O0.csv)

# Filter out torch kernels
if [ -n "${SOUFFLE_RUN}" ] && [ "${SOUFFLE_RUN}" = "TRUE" ]; then
ncu ${NCU_ARGS} -o ncu-souffle_swin_trans_O1 -f \
  python3 souffle_swin_trans.py O1 1 1
fi
ncu -i ./ncu-souffle_swin_trans_O1.ncu-rep --csv --page raw | grep -v "void at::native*" > ncu-souffle_swin_trans_O1.csv
O1_LATENCY=$(python3 ../../extract_ncu_cuda_kernel_latency.py ncu-souffle_swin_trans_O1.csv)

if [ -n "${SOUFFLE_RUN}" ] && [ "${SOUFFLE_RUN}" = "TRUE" ]; then
ncu ${NCU_ARGS} -o ncu-souffle_swin_trans_O2 -f \
  python3 souffle_swin_trans.py O2 1 1
fi
ncu -i ./ncu-souffle_swin_trans_O2.ncu-rep --csv --page raw | grep -v "at::native*" > ncu-souffle_swin_trans_O2.csv
O2_LATENCY=$(python3 ../../extract_ncu_cuda_kernel_latency.py ncu-souffle_swin_trans_O2.csv)

if [ -n "${SOUFFLE_RUN}" ] && [ "${SOUFFLE_RUN}" = "TRUE" ]; then
ncu ${NCU_ARGS} -o ncu-souffle_swin_trans_O3 -f \
  python3 swin_transformer_main.py | tee ncu-souffle_swin_trans_O3.txt 2>&1
ncu -i ./ncu-souffle_swin_trans_O3.ncu-rep --csv --page raw | grep -v "at::native*" > ncu-souffle_swin_trans_O3.csv
O3_LATENCY=$(python3 ../../extract_ncu_cuda_kernel_latency.py ncu-souffle_swin_trans_O3.csv)

if [ -n "${SOUFFLE_RUN}" ] && [ "${SOUFFLE_RUN}" = "TRUE" ]; then
ncu ${NCU_ARGS} -o ncu-souffle_swin_trans_O4 -f \
  python3 souffle_swin_trans.py O4 1 1
fi
ncu -i ./ncu-souffle_swin_trans_O4.ncu-rep --csv --page raw  | grep -v "at::native*" > ncu-souffle_swin_trans_O4.csv
O4_LATENCY=$(python3 ../../extract_ncu_cuda_kernel_latency.py ncu-souffle_swin_trans_O4.csv)


echo "SwinTrans.:", ${O0_LATENCY}, ${O1_LATENCY}, ${O2_LATENCY},\
   ${O3_LATENCY}, ${O4_LATENCY} | tee table4_swin_trans.csv

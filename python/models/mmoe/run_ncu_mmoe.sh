#!/bin/bash
set -x

NCU_ARGS="--metrics dram__bytes_read,gpu__time_duration --clock-control none --target-processes all"

if [ -n "${SOUFFLE_RUN}" ] && [ "${SOUFFLE_RUN}" = "TRUE" ]; then
ncu ${NCU_ARGS} -o ncu-souffle_mmoe_O0 -f \
 python3 souffle_mmoe.py O0 1 1
fi
ncu -i ./ncu-souffle_mmoe_O0.ncu-rep --csv --page raw > ncu-souffle_mmoe_O0.csv
O0_LATENCY=$(python3 ../../extract_ncu_cuda_kernel_latency.py ncu-souffle_mmoe_O0.csv)

# Filter out torch kernels
if [ -n "${SOUFFLE_RUN}" ] && [ "${SOUFFLE_RUN}" = "TRUE" ]; then
ncu ${NCU_ARGS} -o ncu-souffle_mmoe_O1 -f \
 python3 souffle_mmoe.py O1 1 1
fi
ncu -i ./ncu-souffle_mmoe_O1.ncu-rep --csv --page raw > ncu-souffle_mmoe_O1.csv
O1_LATENCY=$(python3 ../../extract_ncu_cuda_kernel_latency.py ncu-souffle_mmoe_O1.csv)

if [ -n "${SOUFFLE_RUN}" ] && [ "${SOUFFLE_RUN}" = "TRUE" ]; then
ncu ${NCU_ARGS}  -o ncu-souffle_mmoe_O2 -f \
 python3 souffle_mmoe.py O2 1 1
fi
ncu -i ./ncu-souffle_mmoe_O2.ncu-rep --csv --page raw | grep -v "at::native*" > ncu-souffle_mmoe_O1.csv
O2_LATENCY=$(python3 ../../extract_ncu_cuda_kernel_latency.py ncu-souffle_mmoe_O1.csv)

if [ -n "${SOUFFLE_RUN}" ] && [ "${SOUFFLE_RUN}" = "TRUE" ]; then
ncu ${NCU_ARGS} -o ncu-souffle_mmoe_O3 -f \
 python3 global_fuse_mmoe.py O3 1 1
fi
ncu -i ./ncu-souffle_mmoe_O3.ncu-rep --csv --page raw | grep -v "at::native*" > ncu-souffle_mmoe_O3.csv
O3_LATENCY=$(python3 ../../extract_ncu_cuda_kernel_latency.py ncu-souffle_mmoe_O3.csv)

O4_LATENCY=${O3_LATENCY}

echo "MMoE:", ${O0_LATENCY}, ${O1_LATENCY}, ${O2_LATENCY},\
   ${O3_LATENCY}, ${O4_LATENCY} | tee table4_mmoe.csv

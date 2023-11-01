#!/bin/bash
set -x
#  --target-processes all
NCU_ARGS="--metrics dram__bytes_read,gpu__time_duration --clock-control none"

if [ -n "${SOUFFLE_RUN}" ] && [ "${SOUFFLE_RUN}" = "TRUE" ]; then
ncu ${NCU_ARGS}  -o ncu-souffle_resnext_O0 -f  \
   python3 run_souffle_resnext.py O0 1 1
fi
ncu -i ./ncu-souffle_resnext_O0.ncu-rep --csv --page raw > ncu-souffle_resnext_O0.csv
RESNEXT_O0_LATENCY=$(python3 ../../extract_ncu_cuda_kernel_latency.py ncu-souffle_resnext_O0.csv)

# Filter out torch kernels
if [ -n "${SOUFFLE_RUN}" ] && [ "${SOUFFLE_RUN}" = "TRUE" ]; then
ncu ${NCU_ARGS}  -o ncu-souffle_resnext_O1 -f  \
   python3 run_souffle_resnext.py O1 1 1
fi
ncu -i ./ncu-souffle_resnext_O1.ncu-rep --csv --page raw | grep -v "at::native*" | grep -v "void at_cuda_detail*" > ncu-souffle_resnext_O1.csv
RESNEXT_O1_LATENCY=$(python3 ../../extract_ncu_cuda_kernel_latency.py ncu-souffle_resnext_O1.csv)


if [ -n "${SOUFFLE_RUN}" ] && [ "${SOUFFLE_RUN}" = "TRUE" ]; then
ncu ${NCU_ARGS} -o ncu-souffle_resnext_O2 -f  \
   python3 run_souffle_resnext.py O2 1 1
fi
ncu -i ./ncu-souffle_resnext_O2.ncu-rep --csv --page raw | grep -v "at::native*" | grep -v "void at_cuda_detail*" > ncu-souffle_resnext_O2.csv
RESNEXT_O2_LATENCY=$(python3 ../../extract_ncu_cuda_kernel_latency.py ncu-souffle_resnext_O2.csv)

RESNEXT_O3_LATENCY=${RESNEXT_O2_LATENCY}
RESNEXT_O4_LATENCY=${RESNEXT_O2_LATENCY}

# echo "RESNEXT:", ${RESNEXT_O0_LATENCY}, ${RESNEXT_O1_LATENCY}, \
#   ${RESNEXT_O2_LATENCY}, ${RESNEXT_O3_LATENCY}, ${RESNEXT_O4_LATENCY} | tee table4_resnext.csv

python3 -c "print('RESNEXT:, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}'.format(${RESNEXT_O0_LATENCY}, ${RESNEXT_O1_LATENCY}, ${RESNEXT_O2_LATENCY}, ${RESNEXT_O3_LATENCY}, ${RESNEXT_O4_LATENCY}))" | tee table4_resnext.csv

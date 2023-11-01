#!/bin/bash
set -x
#  --target-processes all
NCU_ARGS="--metrics dram__bytes_read,gpu__time_duration --clock-control none"

if [ -n "${SOUFFLE_RUN}" ] && [ "${SOUFFLE_RUN}" = "TRUE" ]; then
ncu ${NCU_ARGS}  -o ncu-souffle_efficientnet_O0 -f  \
   python3 souffle_efficientnet.py O0 1 1
fi
ncu -i ./ncu-souffle_efficientnet_O0.ncu-rep --csv --page raw > ncu-souffle_efficientnet_O0.csv
EFFICIENTNET_O0_LATENCY=$(python3 ../../extract_ncu_cuda_kernel_latency.py ncu-souffle_efficientnet_O0.csv)

# Filter out torch kernels
if [ -n "${SOUFFLE_RUN}" ] && [ "${SOUFFLE_RUN}" = "TRUE" ]; then
ncu ${NCU_ARGS}  -o ncu-souffle_efficientnet_O1 -f  \
   python3 souffle_efficientnet.py O1 1 1
fi
ncu -i ./ncu-souffle_efficientnet_O1.ncu-rep --csv --page raw > ncu-souffle_efficientnet_O1.csv
EFFICIENTNET_O1_LATENCY=$(python3 ../../extract_ncu_cuda_kernel_latency.py ncu-souffle_efficientnet_O1.csv)

if [ -n "${SOUFFLE_RUN}" ] && [ "${SOUFFLE_RUN}" = "TRUE" ]; then
ncu ${NCU_ARGS}  -o ncu-souffle_efficientnet_O2 -f  \
   python3 souffle_efficientnet.py O2 1 1
fi
ncu -i ./ncu-souffle_efficientnet_O2.ncu-rep --csv --page raw | grep -v "at::native*" > ncu-souffle_efficientnet_O2.csv
EFFICIENTNET_O2_LATENCY=$(python3 ../../extract_ncu_cuda_kernel_latency.py ncu-souffle_efficientnet_O2.csv)

if [ -n "${SOUFFLE_RUN}" ] && [ "${SOUFFLE_RUN}" = "TRUE" ]; then
ncu ${NCU_ARGS}  -o ncu-souffle_efficientnet_O3 -f --target-processes \
   all python3 souffle_efficientnet.py O3 1 1
fi
ncu -i ./ncu-souffle_efficientnet_O3.ncu-rep --csv --page raw | grep -v "at::native*" > ncu-souffle_efficientnet_O3.csv
EFFICIENTNET_O3_LATENCY=$(python3 ../../extract_ncu_cuda_kernel_latency.py ncu-souffle_efficientnet_O3.csv)

if [ -n "${SOUFFLE_RUN}" ] && [ "${SOUFFLE_RUN}" = "TRUE" ]; then
ncu ${NCU_ARGS}  -o ncu-souffle_efficientnet_O4 -f  \
   python3 souffle_efficientnet.py O4 1 1
fi
ncu -i ./ncu-souffle_efficientnet_O4.ncu-rep --csv --page raw  | grep -v "at::native*" > ncu-souffle_efficientnet_O4.csv
EFFICIENTNET_O4_LATENCY=$(python3 ../../extract_ncu_cuda_kernel_latency.py ncu-souffle_efficientnet_O4.csv)


# echo "EfficientNet:", ${EFFICIENTNET_O0_LATENCY}, ${EFFICIENTNET_O1_LATENCY}, ${EFFICIENTNET_O2_LATENCY},\
#    ${EFFICIENTNET_O3_LATENCY}, ${EFFICIENTNET_O4_LATENCY} | tee table4_efficientnet.csv

python3 -c "print('EfficientNet:, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}'.format(${EFFICIENTNET_O0_LATENCY}, ${EFFICIENTNET_O1_LATENCY}, ${EFFICIENTNET_O2_LATENCY}, ${EFFICIENTNET_O3_LATENCY}, ${EFFICIENTNET_O4_LATENCY}))" | tee table4_efficientnet.csv
#!/bin/bash

script_directory="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${script_directory}
NCU_ARGS="--metrics dram__bytes_read,gpu__time_duration --clock-control none --target-processes all"
# if [ -n "${SOUFFLE_RUN}" ] && [ "${SOUFFLE_RUN}" = "TRUE" ]; then
# ncu ${NCU_ARGS} -o ncu-efficient_se_module_unittest -f  \
#    python3 test_efficientnet_se_module_unittest.py
# fi
ncu -i ./ncu-efficient_se_module_unittest.ncu-rep --csv --page raw \
  | grep -v "at::native" | grep -v "std::enable_if" | grep -v "void gemv2T_kernel_val" \
  > ncu-efficient_se_module_unittest.csv
# Get kernel latency
python3 ${script_directory}/../../python/extract_ncu_cuda_kernel_latency.py ncu-efficient_se_module_unittest.csv
cd ${script_directory}/scripts && python3 draw_performance_breakdown.py
# Generate efficientnet-se-module-latency-ours.pdf


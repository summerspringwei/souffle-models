#!/bin/bash
set -x

NCU_ARGS="--metrics dram__bytes_read,gpu__time_duration --clock-control none --target-processes all"

num_layers=10
num_timesteps=100
if [ -n "${SOUFFLE_RUN}" ] && [ "${SOUFFLE_RUN}" = "TRUE" ]; then
ncu ${NCU_ARGS} -o ncu-souffle_lstm_O0 -f  \
   python3 souffle_lstm.py O0 1 1
fi
ncu -i ./ncu-souffle_lstm_O0.ncu-rep --csv --page raw > ncu-souffle_lstm_O0.csv
LSTM_O0_LATENCY=$(python3 ../../extract_ncu_cuda_kernel_latency.py ncu-souffle_lstm_O0.csv)
LSTM_O0_LATENCY=$(python3 -c "print(${LSTM_O0_LATENCY} * ${num_layers} * ${num_timesteps})")

# Filter out torch kernels
if [ -n "${SOUFFLE_RUN}" ] && [ "${SOUFFLE_RUN}" = "TRUE" ]; then
ncu ${NCU_ARGS} -o ncu-souffle_lstm_O1 -f  \
   python3 souffle_lstm.py O1 1 1
fi
ncu -i ./ncu-souffle_lstm_O1.ncu-rep --csv --page raw | grep -v "at::native*" | grep -v "void at_cuda_detail*" > ncu-souffle_lstm_O1.csv
LSTM_O1_LATENCY=$(python3 ../../extract_ncu_cuda_kernel_latency.py ncu-souffle_lstm_O1.csv)

if [ -n "${SOUFFLE_RUN}" ] && [ "${SOUFFLE_RUN}" = "TRUE" ]; then
ncu ${NCU_ARGS} -o ncu-souffle_lstm_O2 -f  \
   python3 souffle_lstm.py O2 1 1
fi
ncu -i ./ncu-souffle_lstm_O2.ncu-rep --csv --page raw | grep -v "at::native*" | grep -v "void at_cuda_detail*" > ncu-souffle_lstm_O2.csv
LSTM_O2_LATENCY=$(python3 ../../extract_ncu_cuda_kernel_latency.py ncu-souffle_lstm_O2.csv)

if [ -n "${SOUFFLE_RUN}" ] && [ "${SOUFFLE_RUN}" = "TRUE" ]; then
ncu ${NCU_ARGS} -o ncu-souffle_lstm_O3 -f  \
 python3 souffle_lstm.py O3 1 1
fi
ncu -i ./ncu-souffle_lstm_O3.ncu-rep --csv --page raw | grep -v "at::native*" | grep -v "void at_cuda_detail*" > ncu-souffle_lstm_O3.csv
LSTM_O3_LATENCY=$(python3 ../../extract_ncu_cuda_kernel_latency.py ncu-souffle_lstm_O3.csv)

LSTM_O4_LATENCY=${LSTM_O3_LATENCY}

# echo "LSTM:", ${LSTM_O0_LATENCY}, ${LSTM_O1_LATENCY}, \
#   ${LSTM_O2_LATENCY}, ${LSTM_O3_LATENCY}, ${LSTM_O4_LATENCY} | tee table4_lstm.csv

python3 -c "print('LSTM:, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}'.format(${LSTM_O0_LATENCY}, ${LSTM_O1_LATENCY}, ${LSTM_O2_LATENCY}, ${LSTM_O3_LATENCY}, ${LSTM_O4_LATENCY}))" | tee table4_lstm.csv
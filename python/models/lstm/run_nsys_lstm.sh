nsys profile --stats=true -o souffle-report -f true  python3 souffle_lstm.py O0 1 1
sqlite3 --csv souffle-report.sqlite \
    'SELECT names.value AS name, end - start FROM CUPTI_ACTIVITY_KIND_KERNEL AS k JOIN StringIds AS names ON k.demangledName = names.id;' \
    > tmp.csv
O0_LATENCY=$(python3 ../../extract_nsys_cuda_kernel_latency.py tmp.csv)

nsys profile --stats=true -o souffle-report -f true  python3 souffle_lstm.py O1 1 1
sqlite3 --csv souffle-report.sqlite \
    'SELECT names.value AS name, end - start FROM CUPTI_ACTIVITY_KIND_KERNEL AS k JOIN StringIds AS names ON k.demangledName = names.id;' \
    > tmp.csv
O1_LATENCY=$(python3 ../../extract_nsys_cuda_kernel_latency.py tmp.csv)

nsys profile --stats=true -o souffle-report -f true  python3 souffle_lstm.py O2 1 1
sqlite3 --csv souffle-report.sqlite \
    'SELECT names.value AS name, end - start FROM CUPTI_ACTIVITY_KIND_KERNEL AS k JOIN StringIds AS names ON k.demangledName = names.id;' \
    > tmp.csv
O2_LATENCY=$(python3 ../../extract_nsys_cuda_kernel_latency.py tmp.csv)

O3_LATENCY=
O4_LATENCY=${O3_LATENCY}

echo "LSTM:" ${O0_LATENCY} ${O1_LATENCY} ${O2_LATENCY} ${O3_LATENCY} ${O4_LATENCY}

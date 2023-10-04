nsys profile --stats=true -o souffle-report -f true  python3 run_souffle_resnext.py O0 1 1 | tee resnext_O0_tmp.txt 2>&1
sqlite3 --csv souffle-report.sqlite \
    'SELECT names.value AS name, end - start FROM CUPTI_ACTIVITY_KIND_KERNEL AS k JOIN StringIds AS names ON k.demangledName = names.id;' \
    > tmp.csv
RESNEXT_O0_LATENCY=$(python3 ../../extract_nsys_cuda_kernel_latency.py tmp.csv)

nsys profile --stats=true -o souffle-report -f true  python3 run_souffle_resnext.py O1 1 1 | tee resnext_O1_tmp.txt 2>&1
sqlite3 --csv souffle-report.sqlite \
    'SELECT names.value AS name, end - start FROM CUPTI_ACTIVITY_KIND_KERNEL AS k JOIN StringIds AS names ON k.demangledName = names.id;' \
    > tmp.csv
RESNEXT_O1_LATENCY=$(python3 ../../extract_nsys_cuda_kernel_latency.py tmp.csv)

nsys profile --stats=true -o souffle-report -f true  python3 run_souffle_resnext.py O2 1 1 | tee resnext_O2_tmp.txt 2>&1
sqlite3 --csv souffle-report.sqlite \
    'SELECT names.value AS name, end - start FROM CUPTI_ACTIVITY_KIND_KERNEL AS k JOIN StringIds AS names ON k.demangledName = names.id;' \
    > tmp.csv
RESNEXT_O2_LATENCY=$(python3 ../../extract_nsys_cuda_kernel_latency.py tmp.csv)

RESNEXT_O3_LATENCY=${RESNEXT_O2_LATENCY}
RESNEXT_O4_LATENCY=${RESNEXT_O2_LATENCY}

echo "ResNeXt:" ${RESNEXT_O0_LATENCY} ${RESNEXT_O1_LATENCY} ${RESNEXT_O2_LATENCY} ${RESNEXT_O3_LATENCY} ${RESNEXT_O4_LATENCY}

nsys profile --stats=true -o souffle-bert_O0 -f true  python3 souffle_bert.py O0 1 1
sqlite3 --csv souffle-bert_O0.sqlite \
    'SELECT names.value AS name, end - start FROM CUPTI_ACTIVITY_KIND_KERNEL AS k JOIN StringIds AS names ON k.demangledName = names.id;' \
    | grep -vE "at::native|at_cuda_detail" > tmp.csv
O0_LATENCY=$(python3 ../../extract_nsys_cuda_kernel_latency.py tmp.csv)

# nsys profile --stats=true -o souffle-bert_O1 -f true  python3 souffle_bert.py O1 1 1
# sqlite3 --csv souffle-bert_O1.sqlite \
#     'SELECT names.value AS name, end - start FROM CUPTI_ACTIVITY_KIND_KERNEL AS k JOIN StringIds AS names ON k.demangledName = names.id;' \
#     | grep -vE "at::native|at_cuda_detail" > tmp.csv
# O1_LATENCY=$(python3 ../../extract_nsys_cuda_kernel_latency.py tmp.csv)

# nsys profile --stats=true -o souffle-bert_O2 -f true  python3 souffle_bert.py O2 1 1
# sqlite3 --csv souffle-bert_O2.sqlite \
#     'SELECT names.value AS name, end - start FROM CUPTI_ACTIVITY_KIND_KERNEL AS k JOIN StringIds AS names ON k.demangledName = names.id;' \
#     | grep -vE "at::native|at_cuda_detail" > tmp.csv
# O2_LATENCY=$(python3 ../../extract_nsys_cuda_kernel_latency.py tmp.csv)

# nsys profile --stats=true -o souffle-bert_O3 -f true  python3 souffle_bert.py O3 1 1
# sqlite3 --csv souffle-bert_O3.sqlite \
#     'SELECT names.value AS name, end - start FROM CUPTI_ACTIVITY_KIND_KERNEL AS k JOIN StringIds AS names ON k.demangledName = names.id;' \
#     | grep -vE "at::native|at_cuda_detail" > tmp.csv
# O3_LATENCY=$(python3 ../../extract_nsys_cuda_kernel_latency.py tmp.csv)

# nsys profile --stats=true -o souffle-report -f true  python3 souffle_bert.py O4 1 1
# sqlite3 --csv souffle-report.sqlite \
#     'SELECT names.value AS name, end - start FROM CUPTI_ACTIVITY_KIND_KERNEL AS k JOIN StringIds AS names ON k.demangledName = names.id;' \
#     | grep -v "at::native" > tmp.csv
# O4_LATENCY=$(python3 ../../extract_nsys_cuda_kernel_latency.py tmp.csv)

echo "BERT:" ${O0_LATENCY} ${O1_LATENCY} ${O2_LATENCY} ${O3_LATENCY} ${O4_LATENCY}

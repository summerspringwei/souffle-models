# ncu --clock-control none  -o tmp-souffle_swin_trans_O0 -f --target-processes all python3 swin_transformer_naive_main.py O0 1 1
# ncu -i ./tmp-souffle_swin_trans_O0.ncu-rep --csv --page raw > tmp.csv
# O0_LATENCY=$(python3 ../../extract_ncu_cuda_kernel_latency.py tmp.csv)
# echo ${O0_LATENCY}

# Filter out torch kernels
# ncu --clock-control none  -o tmp-souffle_swin_trans_O1 -f --target-processes all python3 souffle_swin_trans.py O1 1 1
# ncu -i ./tmp-souffle_swin_trans_O1.ncu-rep --csv --page raw > tmp.csv
# O1_LATENCY=$(python3 ../../extract_ncu_cuda_kernel_latency.py tmp.csv)
# echo ${O1_LATENCY}

# ncu --clock-control none -o tmp-souffle_swin_trans_O2 -f --target-processes all python3 souffle_swin_trans.py O2 1 1
# ncu -i ./tmp-souffle_swin_trans_O2.ncu-rep --csv --page raw | grep -v "at::native*" > tmp.csv
# O2_LATENCY=$(python3 ../../extract_ncu_cuda_kernel_latency.py tmp.csv)
# echo ${O2_LATENCY}

ncu --clock-control none  -o tmp-souffle_swin_trans_O3 -f --target-processes all python3 swin_transformer_main.py | tee tmp-souffle_swin_trans_O3.txt 2>&1
ncu -i ./tmp-souffle_swin_trans_O3.ncu-rep --csv --page raw | grep -v "at::native*" > tmp.csv
O3_LATENCY=$(python3 ../../extract_ncu_cuda_kernel_latency.py tmp.csv)
echo ${O3_LATENCY}

# ncu --clock-control none  -o tmp-souffle_swin_trans_O4 -f --target-processes all python3 souffle_swin_trans.py O4 1 1
# ncu -i ./tmp-souffle_swin_trans_O4.ncu-rep --csv --page raw  | grep -v "at::native*" > tmp.csv
# O4_LATENCY=$(python3 ../../extract_ncu_cuda_kernel_latency.py tmp.csv)
# echo ${O4_LATENCY}

# echo "EfficientNet:" ${O0_LATENCY} ${O1_LATENCY} ${O2_LATENCY} ${O3_LATENCY} ${O4_LATENCY}

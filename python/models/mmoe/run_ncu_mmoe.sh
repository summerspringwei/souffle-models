ncu --clock-control none --set full -o tmp-souffle_mmoe_O0 -f --target-processes all python3 souffle_mmoe.py O0 1 1
ncu -i ./tmp-souffle_mmoe_O0.ncu-rep --csv --page raw > tmp.csv
O0_LATENCY=$(python3 ../../extract_ncu_cuda_kernel_latency.py tmp.csv)

# Filter out torch kernels
ncu --clock-control none --set full -o tmp-souffle_mmoe_O1 -f --target-processes all python3 souffle_mmoe.py O1 1 1
ncu -i ./tmp-souffle_mmoe_O1.ncu-rep --csv --page raw > tmp.csv
O1_LATENCY=$(python3 ../../extract_ncu_cuda_kernel_latency.py tmp.csv)
echo ${O1_LATENCY}

ncu --clock-control none -o tmp-souffle_mmoe_O2 -f --target-processes all python3 souffle_mmoe.py O2 1 1
ncu -i ./tmp-souffle_mmoe_O2.ncu-rep --csv --page raw | grep -v "at::native*" > tmp.csv
O2_LATENCY=$(python3 ../../extract_ncu_cuda_kernel_latency.py tmp.csv)
echo ${O2_LATENCY}

ncu --clock-control none --set full -o tmp-souffle_mmoe_O3 -f --target-processes all python3 global_fuse_mmoe.py O3 1 1
ncu -i ./tmp-souffle_mmoe_O3.ncu-rep --csv --page raw | grep -v "at::native*" > tmp.csv
O3_LATENCY=$(python3 ../../extract_ncu_cuda_kernel_latency.py tmp.csv)
echo ${O3_LATENCY}

ncu --clock-control none --set full -o tmp-souffle_mmoe_O4 -f --target-processes all python3 souffle_mmoe.py O4 1 1
ncu -i ./tmp-souffle_mmoe_O4.ncu-rep --csv --page raw  | grep -v "at::native*" > tmp.csv
O4_LATENCY=$(python3 ../../extract_ncu_cuda_kernel_latency.py tmp.csv)
echo ${O4_LATENCY}

echo "EfficientNet:" ${O0_LATENCY} ${O1_LATENCY} ${O2_LATENCY} ${O3_LATENCY} ${O4_LATENCY}
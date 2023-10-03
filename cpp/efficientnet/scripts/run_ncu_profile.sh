export HOME=/home/xiachunwei
export TVM_HOME=$HOME/Software/clean_tvm/tvm

export LD_LIBRARY_PATH=$TVM_HOME/build/
export CUDA_HOME="/usr/local/cuda"
export PATH=${CUDA_HOME}/bin/:$PATH
export TORCH_HOME=${HOME}/Software/pytf2.4/lib/python3.7/site-packages/torch
export PYTHONPATH=$TVM_HOME/python:$TORCH_HOME:$PYTHONPATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib:${TORCH_HOME}/lib:$LD_LIBRARY_PATH

# EXEC="../../../101_release/efficientnet_se_module_main 1 1 2 7"
EXEC="../../../../101_release/./efficientnet_se_module_unittest /home/xiachunwei/Projects/EfficientNet-PyTorch/efficientnet-b0.pt"

sudo -E /usr/local/cuda-11.7/bin/ncu  --set full -f --target-processes all -o efficientnet-se_module_v2-max-block --clock-control none -k regex:efficientnet_se_module_v2* $EXEC


export HOME=/home/xiachunwei
export TVM_HOME=$HOME/Software/clean_tvm/tvm

export LD_LIBRARY_PATH=$TVM_HOME/build/
export CUDA_HOME="/usr/local/cuda"
export PATH=${CUDA_HOME}/bin/:$PATH
export TORCH_HOME=${HOME}/Software/pytf2.4/lib/python3.7/site-packages/torch
export PYTHONPATH=$TVM_HOME/python:$TORCH_HOME:$PYTHONPATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib:${TORCH_HOME}/lib:$LD_LIBRARY_PATH
# sudo -E /home/xiachunwei/anaconda3/bin/python3
sudo -E echo $LD_LIBRARY_PATH
sudo -E echo $PYTHONPATH

sudo -E /usr/local/cuda/bin//ncu --mode=launch /home/xiachunwei/Software/anaconda3/bin/python3 tvm_resnext.py

# sudo /usr/local/cuda/bin//ncu --mode=attach --hostname 127.0.0.1 --set full -o resnext-101-apollo-1410 -f --clock-control none
# sudo /usr/local/cuda/bin/ncu --metrics regex:sm__inst_executed* -k fused_sqq_bert_attn -o fused_sqq_bert_attn_opcodes -f --target-processes all ./torch_bert_attn_sqq 1 1 13

import logging
import torch
import pandas as pd
import numpy as np
from torch import nn
from transformers import BertForQuestionAnswering, BertTokenizer, BertModel

import bert_binding

logging.basicConfig(level=logging.INFO)


"""Test the correctness of the souffle bert layer with self implemented bert layer
Things to consider:
1. Layout:
Note: for the Linear layer, we use the GEMM representation C = A(M, K) * B(K, N).
The default layout for torch weight is [out_features (N), in_features (K)]
I have marked the layout of weights in the code.
For the layerout of the intermediate tensor, please refer to bert_binding.cu:32-78 to check the layout.
Refer to this link for details: https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html

2. Mask:
We did't implement the mask operation in the self implemented bert layer and the souffle bert layer.
If you need mask please modify the corresponding CUDA code.

3. Test:
Please use `torch.testing.assert_close` to check the correctness of the outputs.

"""

class BertLayerModule(nn.Module):
    """Self implemented bert layer"""
    def __init__(self, batch_size, max_seq_length, num_heads, hidden_size, device=torch.device('cuda'), dtype=torch.float16, fused_qkv=False) -> None:
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.fused_qkv = fused_qkv

        self.d_model = num_heads * hidden_size
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.num_heads = num_heads
        self.hidden_size = hidden_size
    
  
    def init_weights_with_bert(self, original_model, layer_idx = 5):
        # init with a layer
        layer = original_model.encoder.layer[layer_idx]
        attention = layer.attention.self
        q_weight = attention.query.weight.data.clone()  # [N768, K768]
        k_weight = attention.key.weight.data.clone()  # [N768, K768]
        v_weight = attention.value.weight.data.clone()  # [N768, K768]
        attn_output_weight = layer.attention.output.dense.weight.data.clone()  # [N768, K768]
        self.ff_fc1_weight = layer.intermediate.dense.weight.data.clone()  # [3072, 768]
        self.ff_fc2_weight = layer.output.dense.weight.data.clone()  # [768, 3072]

        self.qkv_matmul_weight = torch.concat([q_weight, k_weight, v_weight], dim=0).to(torch.float16).to(self.device) # [3N768, K768]
        self.fc_weight = attn_output_weight.to(self.dtype).to(self.device) # [N768, K768]

        # Set up Linear parameters
        factory_kwargs = {'device': self.device, 'dtype': self.dtype}
        self.qkv_matmul = nn.Linear(self.d_model, self.d_model * 3, **factory_kwargs)
        self.qkv_matmul.weight = nn.Parameter(self.qkv_matmul_weight)
        self.qkv_matmul.bias = nn.Parameter(torch.zeros(self.d_model * 3, **factory_kwargs))
        self.fc = nn.Linear(self.d_model, self.d_model, **factory_kwargs)
        self.fc.weight = nn.Parameter(self.fc_weight)
        self.fc.bias = nn.Parameter(torch.zeros(self.d_model, **factory_kwargs))
        self.ff_fc1 = nn.Linear(self.d_model, self.d_model * 4, **factory_kwargs)
        self.ff_fc1.weight = nn.Parameter(self.ff_fc1_weight)
        self.ff_fc1.bias = nn.Parameter(torch.zeros(self.d_model * 4, **factory_kwargs))
        self.ff_fc2 = nn.Linear(self.hidden_size * 4, self.d_model, **factory_kwargs)
        self.ff_fc2.weight = nn.Parameter(self.ff_fc2_weight)
        self.ff_fc2.bias = nn.Parameter(torch.zeros(self.d_model, **factory_kwargs))
        # Dump the parameters to csv
        # pd.DataFrame(self.fc_weight.detach().cpu().numpy()).to_csv("debug_outputs/fc_weight.csv", index=False, header=False)

  
    def forward(self, src: torch.Tensor):
        if self.fused_qkv:
        t_output_qkv = self.qkv_matmul(src)
        qkv = torch.split(t_output_qkv, 768, 2)
        q = qkv[0].reshape(self.batch_size, self.max_seq_length, self.num_heads, self.hidden_size).permute(0, 2, 1, 3)
        k = qkv[1].reshape(self.batch_size, self.max_seq_length, self.num_heads, self.hidden_size).permute(0, 2, 1, 3)
        v = qkv[2].reshape(self.batch_size, self.max_seq_length, self.num_heads, self.hidden_size).permute(0, 2, 1, 3)
        return_t_output_qkv = torch.concat([q, k, v], dim=0) # (batch_size * 3, num_heads, max_seq_length, hidden_size)
        t_query = torch.permute(torch.reshape(qkv[0], (self.batch_size*self.max_seq_length, self.num_heads, self.hidden_size)), (1, 0, 2)) # (num_heads, max_seq_length, hidden_size)
        t_key = torch.permute(torch.reshape(qkv[1], (self.batch_size*self.max_seq_length, self.num_heads, self.hidden_size)), (1, 2, 0)) # (num_heads, hidden_size, max_seq_length)
        t_value = torch.permute(torch.reshape(qkv[2], (self.batch_size*self.max_seq_length, self.num_heads, self.hidden_size)), (1, 0, 2)) # (num_heads, max_seq_length, hidden_size)
        else:
        t_query = torch.permute(torch.reshape(self.query(src), (self.batch_size*self.max_seq_length, self.num_heads, self.hidden_size)), (1, 0, 2))
        t_key = torch.permute(torch.reshape(self.key(src), (self.batch_size*self.max_seq_length, self.num_heads, self.hidden_size)), (1, 2, 0)) # (num_heads, hidden_size, max_seq_length)
        t_value = torch.permute(torch.reshape(self.value(src), (self.batch_size*self.max_seq_length, self.num_heads, self.hidden_size)), (1, 0, 2)) # (num_heads, max_seq_length, hidden_size)
        factor = torch.tensor(((self.hidden_size * self.num_heads) ** 0.5, ), dtype=self.dtype).cuda()
        t_query_key_output = torch.softmax(torch.divide(torch.bmm(t_query, t_key), factor), 2)
        t_attn_value_output = torch.bmm(t_query_key_output, t_value) # (num_heads, max_seq_length, hidden_size)
        t_attn_value_output_permuted = torch.reshape(torch.permute(t_attn_value_output, (1, 0, 2)), (self.batch_size * self.max_seq_length, self.d_model))
        t_attn_fc_output_tmp = self.fc(t_attn_value_output_permuted)
        t_attn_fc_output = torch.add(t_attn_fc_output_tmp, src)
        t_attn_layer_norm_output = torch.layer_norm(t_attn_fc_output, (self.d_model,)).reshape(self.batch_size* self.max_seq_length, self.d_model)
        t_ff_fc1_output = torch.nn.functional.relu(self.ff_fc1(t_attn_layer_norm_output))
        # pd.DataFrame(t_attn_layer_norm_output.detach().cpu().numpy()).to_csv("debug_outputs/torch_attn_layer_norm_output.csv", index=False, header=False)
        # pd.DataFrame(self.ff_fc1.weight.detach().cpu().numpy()).to_csv("debug_outputs/torch_ff_fc1_weight.csv", index=False, header=False)
        # pd.DataFrame(t_ff_fc1_output.detach().cpu().numpy()).to_csv("debug_outputs/torch_ff_fc1_output.csv", index=False, header=False)
        t_ff_fc2_output = self.ff_fc2(t_ff_fc1_output)
        t_ff_output = torch.add(t_ff_fc2_output, t_attn_layer_norm_output)
        t_output = torch.layer_norm(t_ff_output, (self.d_model,))
        # pd.DataFrame(t_attn_value_output_permuted.detach().cpu().numpy()).to_csv("debug_outputs/torch_attn_value_output_permuted.csv", index=False, header=False)


        return return_t_output_qkv, t_query_key_output,  t_attn_value_output, t_attn_value_output_permuted, t_attn_fc_output_tmp, t_attn_layer_norm_output, t_ff_fc1_output, t_ff_fc2_output, t_ff_output, t_output


class SouffleBertLayerModule(nn.Module):
    def __init__(self, batch_size, max_seq_length, num_heads, hidden_size, device=torch.device('cuda'), dtype=torch.float16, fused_qkv=False) -> None:
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.fused_qkv = fused_qkv

        self.d_model = num_heads * hidden_size
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.qkv_weight = None
        self.fc_weight = None

    
    def init_weights_with_bert(self, original_model, layer_idx = 5):
        factory_kwargs = {'device': self.device, 'dtype': self.dtype}
        # init with a layer
        layer = original_model.encoder.layer[layer_idx]
        attention = layer.attention.self
        q_weight = attention.query.weight.data.clone()  # [N768, K768]
        k_weight = attention.key.weight.data.clone()  # [N768, K768]
        v_weight = attention.value.weight.data.clone()  # [N768, K768]
        attn_output_weight = layer.attention.output.dense.weight.data.clone()  # [N768, K768]

        # !Important: Set up weights, please note how to transpose the weights to match the cuda kernel
        self.qkv_weight = torch.stack([q_weight.t().contiguous(), k_weight.t().contiguous(), v_weight.t().contiguous()], dim=0).to(self.dtype).to(self.device) # [3K768, N768]
        self.fc_weight = attn_output_weight.t().contiguous().to(torch.float16).to(self.device) # [K768, N768] should be K * N in cuda kernel
        self.ff_fc1_weight = layer.intermediate.dense.weight.data.clone().to(torch.float16).to(self.device).t().contiguous()  # [K768, N3072]
        self.ff_fc2_weight = layer.output.dense.weight.data.clone().to(torch.float16).to(self.device).t().contiguous()  # [K3072, N768]


    def forward(self, src):
        output_qkv, query_key_output, attn_value_output, attn_fc_output, feed_forward_fc1_output, feed_forward_fc2_output = bert_binding.souffle_bert_layer(
            src,
            self.qkv_weight,
            self.fc_weight,
            self.ff_fc1_weight,
            self.ff_fc2_weight,
            3  # opt_level
        )
        return output_qkv, query_key_output, attn_value_output, attn_fc_output, feed_forward_fc1_output, feed_forward_fc2_output


def load_model_to_cuda(model_name = "bert-base-uncased"):
    # 加载原始模型
    print(f"Loading model: {model_name}")
    original_model = BertModel.from_pretrained(model_name, torch_dtype=torch.float16)
    original_model.eval()
    if torch.cuda.is_available():
        original_model = original_model.cuda()

    return original_model


def prepare_input(original_model):
    # 加载原始模型
    model_name = "bert-base-uncased"
    print(f"Loading model: {model_name}")
    tokenizer = BertTokenizer.from_pretrained(model_name, torch_dtype=torch.float16)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    text = """The actual opcode is VDPBF16PS (vector dot-product bf16 → packed single-precision). It computes dot-products of bf16 lanes and accumulates into float32 results in zmm registers.
	•	Compilers may expose this via intrinsics such as _mm512_dpbf16_ps or _mm512_dpbf16_ps_* (names vary by vendor/headers). If available, you would:
	1.	Load 32 bf16 elements (or 64) packed into a zmm-like bf16 container (toolchains define __m512bh for bf16 vectors),
	2.	Use the dpbf16 intrinsic to do the dot product and accumulate into __m512 float accumulators.
	•	If your compiler doesn’t yet have that intrinsic, you can emit vdpbf16ps in inline asm. Syntax and operand ordering depends on assembler (AT&T vs Intel syntax), toolchain, and exact operand types; check your assembler/intrinsics reference."""
    inputs = tokenizer(text, return_tensors='pt', max_length=384, truncation=True, padding='max_length')
    if torch.cuda.is_available():
        inputs = {k: v.to('cuda') for k, v in inputs.items()}
    with torch.no_grad():
        input_embeddings = original_model.embeddings(
            input_ids=inputs["input_ids"],
            token_type_ids=inputs.get("token_type_ids"),
        ).to(device)

    return input_embeddings


def run_bert_layer_test():
    original_model = load_model_to_cuda()

    bert_layer_module = BertLayerModule(batch_size=1, max_seq_length=384, num_heads=12, hidden_size=64, device='cuda', dtype=torch.float16, fused_qkv=True)
    bert_layer_module.init_weights_with_bert(original_model, layer_idx=5)

    souffle_bert_layer_module = SouffleBertLayerModule(batch_size=1, max_seq_length=384, num_heads=12, hidden_size=64, device='cuda', dtype=torch.float16, fused_qkv=True)
    souffle_bert_layer_module.init_weights_with_bert(original_model, layer_idx=5)

    input_embeddings = prepare_input(original_model)

    with torch.no_grad():
        output_qkv, query_key_output,  t_attn_value_output, t_attn_value_output_permuted, t_attn_fc_output_tmp, t_attn_layer_norm_output, t_ff_fc1_output, t_ff_fc2_output, t_ff_output, t_output = bert_layer_module(input_embeddings)
        souffle_output_qkv, souffle_query_key_output, souffle_attn_value_output, souffle_attn_fc_output, souffle_feed_forward_fc1_output, souffle_feed_forward_fc2_output = souffle_bert_layer_module(input_embeddings)
        torch.cuda.synchronize()
        torch.testing.assert_close(souffle_output_qkv, output_qkv, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(souffle_query_key_output, query_key_output, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(souffle_attn_value_output, t_attn_value_output_permuted, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(souffle_attn_fc_output, t_attn_layer_norm_output, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(souffle_feed_forward_fc1_output, t_ff_fc1_output, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(souffle_feed_forward_fc2_output, t_output, rtol=1e-2, atol=2e-2)
        print("All outputs match between BertLayerModule and SouffleBertLayerModule!")


if __name__ == "__main__":
    run_bert_layer_test()

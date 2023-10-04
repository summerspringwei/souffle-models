import os
import torch
from torch import nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np


def run_transformer():
  # Bert base, 12-layer, 768-hidden, 12-heads, 110M parameters
  transformer_model = nn.Transformer(d_model=768, nhead=12, num_encoder_layers=12, dim_feedforward=3072)
  src = torch.rand((1, 512, 768))
  tgt = torch.rand((1, 512, 768))
  out = transformer_model(src, tgt)


class AttentionModule(nn.Module):
  def __init__(self, batch_size, max_seq_length, num_heads, hidden_size, device=torch.device('cuda'), dtype=torch.float16, fused_qkv=False) -> None:
    super().__init__()
    factory_kwargs = {'device': device, 'dtype': dtype}
    self.dtype = dtype
    self.fused_qkv = fused_qkv
    self.d_model = num_heads * hidden_size
    self.batch_size = batch_size
    self.max_seq_length = max_seq_length
    self.num_heads = num_heads
    self.hidden_size = hidden_size
    self.qkv_matmul = nn.Linear(self.d_model, self.d_model * 3, **factory_kwargs)
    self.fc = nn.Linear(self.d_model, self.d_model, **factory_kwargs)
    self.query = nn.Linear(self.d_model, self.d_model, **factory_kwargs)
    self.key = nn.Linear(self.d_model, self.d_model, **factory_kwargs)
    self.value = nn.Linear(self.d_model, self.d_model, **factory_kwargs)

  
  def forward(self, src: torch.Tensor):
    if self.fused_qkv:
      t_output_qkv = self.qkv_matmul(src)
      qkv = torch.split(t_output_qkv, 768, 1)
      t_query = torch.permute(torch.reshape(qkv[0], (self.batch_size*self.max_seq_length, self.num_heads, self.hidden_size)), (1, 0, 2)) # (num_heads, max_seq_length, hidden_size)
      t_key = torch.permute(torch.reshape(qkv[1], (self.batch_size*self.max_seq_length, self.num_heads, self.hidden_size)), (1, 2, 0)) # (num_heads, hidden_size, max_seq_length)
      t_value = torch.permute(torch.reshape(qkv[2], (self.batch_size*self.max_seq_length, self.num_heads, self.hidden_size)), (1, 0, 2)) # (num_heads, max_seq_length, hidden_size)
    else:
      t_query = torch.permute(torch.reshape(self.query(src), (self.batch_size*self.max_seq_length, self.num_heads, self.hidden_size)), (1, 0, 2))
      t_key = torch.permute(torch.reshape(self.key(src), (self.batch_size*self.max_seq_length, self.num_heads, self.hidden_size)), (1, 2, 0)) # (num_heads, hidden_size, max_seq_length)
      t_value = torch.permute(torch.reshape(self.value(src), (self.batch_size*self.max_seq_length, self.num_heads, self.hidden_size)), (1, 0, 2)) # (num_heads, max_seq_length, hidden_size)
    factor = torch.tensor((self.hidden_size, ), dtype=self.dtype).cuda()
    t_query_key_output = torch.softmax(torch.divide(torch.bmm(t_query, t_key), factor), 2)
    t_attn_value_output = torch.bmm(t_query_key_output, t_value)
    # t_attn_value_output_permuted = torch.reshape(torch.permute(t_attn_value_output, (0, 2, 1, 3)), (self.batch_size, self.max_seq_length, self.d_model))
    t_attn_value_output_permuted = torch.reshape(torch.permute(t_attn_value_output, (1, 0, 2)), (self.batch_size, self.max_seq_length, self.d_model))
    t_attn_fc_output_tmp = self.fc(t_attn_value_output_permuted)
    t_attn_fc_output = torch.add(t_attn_fc_output_tmp, src)
    t_attn_layer_norm_output = torch.layer_norm(t_attn_fc_output, (self.d_model,))
    
    return t_attn_layer_norm_output
  

  def _initialize_weights(self):
    init.orthogonal_(self.qkv_matmul.weight, init.calculate_gain('relu'))
    init.orthogonal_(self.fc.weight, init.calculate_gain('relu'))
    init.orthogonal_(self.query.weight, init.calculate_gain('relu'))
    init.orthogonal_(self.key.weight, init.calculate_gain('relu'))
    init.orthogonal_(self.value.weight, init.calculate_gain('relu'))

  

def freeze_attn(batch_size, max_seq_length, num_heads, hidden_size):
  d_model = num_heads * hidden_size
  model_name = "bert_attn_{}_{}_{}_{}".format(batch_size, max_seq_length, num_heads, hidden_size)
  src = torch.randn((batch_size * max_seq_length, d_model), requires_grad=False, dtype=torch.float16).cuda()
  attn_module = AttentionModule(batch_size, max_seq_length, num_heads, hidden_size)
  t_output = attn_module(src)
  torch.onnx.export(attn_module,               # model being run
                    src,                         # model input (or a tuple for multiple inputs)
                    "{}.onnx".format(model_name),   # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=10,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = ['input'],   # the model's input names
                    output_names = ['output'], # the model's output names
                    # dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                    #               'output' : {0 : 'batch_size'}}
                    )




class FeedForward(nn.Module):
  def __init__(self, d_model, dim_feedforward, dropout=0.0, layer_norm_eps=1e-5, device=None, dtype=None):
    super().__init__()
    factory_kwargs = {'device': device, 'dtype': dtype}
    self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
    self.dropout = nn.Dropout(dropout)
    self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)
    self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
    self.activation = F.relu
  
  def forward(self, src: torch.Tensor):
    fc1_output = self.dropout(self.activation(self.linear1(src)))
    print(fc1_output.shape)
    src2 = self.linear2(fc1_output)
    src = src + self.dropout(src2)
    src = self.norm2(src)
    return src

  def _initialize_weights(self):
    init.orthogonal_(self.linear1.weight, init.calculate_gain('relu'))
    init.orthogonal_(self.linear2.weight, init.calculate_gain('relu'))


def freeze_feed_forward():
  # Input to the model
  batch_size, seq_length, d_model, dim_feedforward = 1, 384, 768, 3072
  model_name = "bert_feed_forward_{}_{}_{}_{}".format(batch_size, seq_length, d_model, dim_feedforward)
  src = torch.randn((batch_size, seq_length, d_model), requires_grad=False).cuda().to(torch.float16)
  feed_forward_model = FeedForward(d_model, dim_feedforward, device=torch.device("cuda"), dtype=torch.float16)
  torch_out = feed_forward_model(src)

  # Export the model
  torch.onnx.export(feed_forward_model,               # model being run
                    src,                         # model input (or a tuple for multiple inputs)
                    "{}.onnx".format(model_name),   # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=10,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = ['input'],   # the model's input names
                    output_names = ['output'], # the model's output names
                    dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                  'output' : {0 : 'batch_size'}})


class BERT(nn.Module):
  def __init__(self, batch_size, max_seq_length, num_heads, hidden_size, dim_feedforward, device=torch.device('cuda'), dtype=torch.float16, fused_qkv=False):
    super().__init__()
    self.attn = AttentionModule(batch_size, max_seq_length, num_heads, hidden_size, fused_qkv=fused_qkv)
    self.feed_forward = FeedForward(num_heads*hidden_size, dim_feedforward, device=device, dtype=dtype)

  def forward(self, src):
    attn_output = self.attn(src)
    print(attn_output.shape)
    # attn_output = torch.reshape(attn_output, (384, 768))
    module_output = self.feed_forward(attn_output)
    return module_output


def freeze_bert():
  batch_size, max_seq_length, d_model, dim_feedforward = 1, 384, 768, 3072
  num_heads, hidden_size = 12, 64
  model_name = "bert_{}_{}_{}_{}".format(batch_size, max_seq_length, d_model, dim_feedforward)
  src = torch.randn((batch_size, max_seq_length, d_model), requires_grad=False).cuda().to(torch.float16)
  bert_model = BERT(batch_size, max_seq_length, num_heads, hidden_size, dim_feedforward)
  torch_out = bert_model(src)

    # Export the model
  torch.onnx.export(bert_model,               # model being run
                    src,                         # model input (or a tuple for multiple inputs)
                    "{}.onnx".format(model_name),   # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=10,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = ['input'],   # the model's input names
                    output_names = ['output'], # the model's output names
                    dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                  'output' : {0 : 'batch_size'}})


def convert_pb_2_onnx(model_folder, model_name):
  sh_cmd = "python3 -m tf2onnx.convert --input {} --output {} --inputs {} --outputs {}".format(\
      os.path.join(model_folder, model_name+".pb"), os.path.join(model_folder, "{}.onnx".format(model_name)),\
        "", "layer_11/output/LayerNorm/batchnorm/add_1:0")
  
  os.system(sh_cmd)


if __name__=="__main__":
  # freeze_feed_forward()
  # freeze_attn(1, 128, 12, 64)
  # freeze_bert()
  convert_pb_2_onnx("/home/xiachunwei/Software/fusion/frozen_pbs", "bert-lyj")

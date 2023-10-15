import sys

import torch
import bert_binding
import souffle_bert_base

batch_size, num_heads, max_seq_length, num_hidden, d_intermedia = 1, 12, 384, 64, 3072
hidden_size = num_heads * num_hidden
def test_bert_binding_one_layer(opt_level: int):
  src = torch.randn(batch_size, max_seq_length, hidden_size).to(torch.float16).to('cuda')
  # src_mask = torch.ones(batch_size, max_seq_length, dtype=torch.int32)
  qkv_weight = torch.randn(3, hidden_size, hidden_size).to(torch.float16).to('cuda')
  attn_fc_weight = torch.randn(hidden_size, hidden_size).to(torch.float16).to('cuda')
  feed_forward_fc1_weight = torch.randn(hidden_size, d_intermedia).to(torch.float16).to('cuda')
  feed_forward_fc2_weight = torch.randn(d_intermedia, hidden_size).to(torch.float16).to('cuda')
  output = bert_binding.souffle_bert_layer(src, qkv_weight, attn_fc_weight, feed_forward_fc1_weight, feed_forward_fc2_weight, opt_level)
  print(output)


def main():  
  opt_level, num_bench, num_repeat = "O2", 1, 1
  # Parse arguments
  if len(sys.argv) <= 1:
      print("Usage: python3 run_souffle_resnext.py [opt_level]")
  opt_level = str(sys.argv[1])
  if len(sys.argv) > 2:
      num_bench = int(sys.argv[2])
  if len(sys.argv) > 3:
      num_repeat = int(sys.argv[3])
  if opt_level == "O0":
     souffle_bert_base.souffle_bert_base()
  if opt_level == "O1":
    test_bert_binding_one_layer(1)
  elif opt_level == "O2":
    test_bert_binding_one_layer(2)
  elif opt_level == "O3":
    test_bert_binding_one_layer(3)
  elif opt_level == "O4":
    test_bert_binding_one_layer(4)


if __name__ == '__main__':
  main()

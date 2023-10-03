import torch
import bert_binding

batch_size, num_heads, max_seq_length, hidden_size, d_intermedia = 1, 12, 384, 64, 3072

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
  test_bert_binding_one_layer(0)
  test_bert_binding_one_layer(1)
  test_bert_binding_one_layer(2)
  test_bert_binding_one_layer(3)

if __name__ == '__main__':
  main()

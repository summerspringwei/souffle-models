import os, sys

from swin_transformer_naive_main import SwinTransformerNaive
from swin_transformer_main import TVMSwinTransformer
from swin_transformer_with_cutlass_main import TVMSwinTransformerO3

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
    model = SwinTransformerNaive(hfuse=False, tune=False, num_trials=3000, num_bench=1, num_repeats=1)
    model.forward()
  elif opt_level == "O1":
    model = SwinTransformerNaive(hfuse=True, tune=False, num_trials=3000, num_bench=1, num_repeats=1)
    model.forward()
  elif opt_level == "O2":
    model = TVMSwinTransformerO3()
    model.forward()
  elif opt_level == "O3":
    model = TVMSwinTransformerO3()
    model.forward()
  elif opt_level == "O4":
    model = TVMSwinTransformerO3()
    model.forward()

if __name__=="__main__":
  main()

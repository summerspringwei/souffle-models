import sys

from tvm_synthetic_mmoe import FusedMMoE
from hori_mmoe import HoriMMoE
from naive_mmoe import NaiveMMoE

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
    model = NaiveMMoE(1, 100, 16, 8, 2, tune=False, num_trials=20, num_bench=num_bench, num_repeats=num_repeat)
    model.forward()
  elif opt_level == "O1":
    model = HoriMMoE(1, 100, 16, 8, 2, tune=False, num_trials=100, num_bench=num_bench, num_repeats=num_repeat)
    model.forward() 
  elif opt_level == "O2":
    model = FusedMMoE(1, 100, 16, 8, 2, tune=False, num_trails=20, num_bench=num_bench, num_repeats=num_repeat)
    model.forward() 
  elif opt_level == "O3":
    pass
  elif opt_level == "O4":
    pass


if __name__ == "__main__":
  main()
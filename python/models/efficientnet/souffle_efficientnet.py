import sys

from tvm_efficientnet import TVMEfficientNet
from naive_efficientnet import NaiveTVMEfficientNet

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
    # Native implementation, without fusion
    if opt_level == "O0":
      model = NaiveTVMEfficientNet("efficientnet-b0", se_mode=0, tune=False, num_trials=400, num_bench=num_bench, num_repeats=num_repeat)
      model.forward()
    # Horizontal fusion
    elif opt_level == "O1":
      model = NaiveTVMEfficientNet("efficientnet-b0", se_mode=0, tune=False, num_trials=400, num_bench=num_bench, num_repeats=num_repeat)
      model.forward()
    # Vertical fusion
    elif opt_level == "O2":
      model = TVMEfficientNet("efficientnet-b0", 0, num_bench=num_bench, num_repeat=num_repeat)
      model.forward()
    # Global sync
    elif opt_level == "O3":
      model = TVMEfficientNet("efficientnet-b0", 2, num_bench=num_bench, num_repeat=num_repeat)
      model.forward()
    # Global optimization
    elif opt_level == "O4":
      model = TVMEfficientNet("efficientnet-b0", 3, num_bench=num_bench, num_repeat=num_repeat)
      model.forward()

if __name__=="__main__":
  main()

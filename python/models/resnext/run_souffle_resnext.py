import sys

from hori_fusion_resnext import ResNextHorizontal
from souffle_resnext import ResNext

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
    
    # Run model
    if opt_level == "O0":   
        model = ResNext(224, [3, 4, 23, 3], fuse=False, num_bench=num_bench, num_repeat=num_repeat)
        model.forward()
    elif opt_level == "O1":
        model = ResNextHorizontal(224, [3, 4, 23, 3], tune=False, num_trials=2000, num_bench=num_bench, num_repeats=num_repeat)
        model.forward()
    elif opt_level == "O2":
        model = ResNext(224, [3, 4, 23, 3], fuse=True, num_bench=num_bench, num_repeat=num_repeat)
        model.forward()
    elif opt_level == "O3":
        model = ResNext(224, [3, 4, 23, 3], fuse=True, num_bench=num_bench, num_repeat=num_repeat)
        model.forward()
    elif opt_level == "O4":
        model = ResNext(224, [3, 4, 23, 3], fuse=True, num_bench=num_bench, num_repeat=num_repeat)
        model.forward()
    else:
        print("Invalid opt level")
        exit(1)


if __name__ == "__main__":
    main()
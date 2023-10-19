
import efficientnet_se_module_unittest_binding

"""Usage:
python3 test_efficientnet_se_module_unittest.py
"""

def main():
  ret = efficientnet_se_module_unittest_binding.run_all_se_module_unittest("efficientnet-b0.pt")
  assert(ret==0)


if __name__ == "__main__":
  main()
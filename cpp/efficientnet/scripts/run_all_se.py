
import os

repeat, loop, is_pipe = 1, 1, 0
for func in range(9):
  # sh_cmd = "../../../release/efficientnet_se_module_main {} {} {} {}".format(repeat, loop, 0, func)
  # os.system(sh_cmd)
  sh_cmd = "../../../release/efficientnet_se_module_main {} {} {} {}".format(repeat, loop, 1, func)
  os.system(sh_cmd)


import re
import xlrd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import matplotlib



matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def rgb_to_hex(r, g, b):
  color = '#{:02x}{:02x}{:02x}'.format(r, g, b)
  print(color)
  return color
# rgb_to_hex(183, 221, 232)
# rgb_to_hex(235, 241, 223)
# rgb_to_hex(252, 235, 221)
# rgb_to_hex(238, 234, 242)
# rgb_to_hex(255, 41, 41)

def read_ncu_xls(file_path):
    book = xlrd.open_workbook(file_path)
    sh = book.sheet_by_index(0)
    print("{0} {1} {2}".format(sh.name, sh.nrows, sh.ncols))
    group_size = 12
    data_groups = []
    start = 1
    end = start + group_size
    while end <= sh.nrows:
      latency_slice = sh.col_values(9, start, end)
      print(latency_slice)
      data_groups.append(latency_slice)
      start += group_size
      end += group_size
    return data_groups


def read_ncu_csv(file_path):
    """
    csv file format:
    kernel_name,latency
    """
    latency_pattern=r',\d+\.?\d*$'
    data_groups = []
    group_size = 12
    all_latency = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
          line = line.strip()
          latency_matches = re.findall(latency_pattern, line)
          if latency_matches:
             latency = float(latency_matches[0][1:])
             all_latency.append(latency)
    
    # reshape the data to n * 12
    data_groups = np.array(all_latency).reshape(-1, group_size).tolist()

    return data_groups


def total_latency(latency_list):
  sum = 0
  sum += (latency_list[0] + latency_list[1] + latency_list[2] * 2 + latency_list[3] * 2 + latency_list[4] * 3 + latency_list[5] * 3 + latency_list[6] * 4)
  return sum


def draw_bars(species, fusion_latency, fig_name, max_ylim):
    x = np.arange(len(species))
    width = 0.2
    multiplier = 0
    font_size = 14
    font = font_manager.FontProperties(family='monospace',
                                  #  weight='bold',
                                   style='normal', size=font_size)
    font_dict = {'family':'monospace','color':'black','size':font_size}
    fig, ax = plt.subplots(layout='constrained', figsize=(9, 3))
    idx = 0
    hatch = ['//', '..', 'xx', '-']
    colors= ["#b7dde8", "#ebf1df", "#fcebdd", "#eeeaf2"]
    for attr, measurement in fusion_latency.items():
      offset = width * multiplier
      rects = ax.bar(x + offset, measurement, width, label=attr, hatch=hatch[idx], color=colors[idx], edgecolor='#000000')
      idx += 1
      # rects = ax.bar(x + offset, width, label=attr)
      # ax.bar_label(rects, padding=3)
      multiplier += 1

    ax.set_ylabel("Speedup", fontname="monospace", fontdict=font_dict)
    # ax.set_title("EfficientNet-b0 swish module")
    for tick in ax.get_xticklabels():
      tick.set_fontname("monospace")
    ax.set_xticks(x+width, prop=font)
    ax.set_xticklabels(species, rotation=0, fontsize=font_size)
    ax.legend(loc='upper center', prop=font, ncol=4)
    ax.set_ylim(0, max_ylim)
    plt.yticks(fontsize=font_size)
    plt.savefig(fig_name)


def draw_efficientnet_plot(group_data, normalize=False):
    unfused_list, tvm_fused_list, one_kernel_list, fused_list = [], [], [], []
    for group in group_data:
        group = np.array(group)
        unfused = np.sum(group[0:7])
        tvm_fused = group[0] + group[6] + group[7] + group[8]
        one_kernel = group[9]
        fused = group[11]
        unfused_list.append(unfused)
        tvm_fused_list.append(tvm_fused)
        one_kernel_list.append(one_kernel)
        fused_list.append(fused)
        print("{} {} {} {}".format(unfused, tvm_fused, one_kernel, fused))
    species = []
    for i in range(len(group_data)):
       species.append("M{}".format(i))
    unfused_list = np.array(unfused_list)
    tvm_fused_list = np.array(tvm_fused_list)
    one_kernel_list = np.array(one_kernel_list)
    fused_list = np.array(fused_list)
    unfused_list_latency = unfused_list
    print(total_latency(unfused_list), total_latency(tvm_fused_list), total_latency(one_kernel_list), total_latency(fused_list))
    if normalize:
       unfused_list = unfused_list_latency / unfused_list
       tvm_fused_list = unfused_list_latency / tvm_fused_list
       one_kernel_list = unfused_list_latency / one_kernel_list
       fused_list  = unfused_list_latency / fused_list
       unfused_list = np.append(unfused_list, 1)
       tvm_fused_list = np.append(tvm_fused_list, np.average(tvm_fused_list))
       print("speedup", tvm_fused_list[-1])
       one_kernel_list = np.append(one_kernel_list, np.average(one_kernel_list))
       print("speedup", one_kernel_list[-1])
       fused_list = np.append(fused_list, np.average(fused_list))
       print("speedup", fused_list[-1])
       species = np.append(species, "AVG")

    # fusion_latency = {
    #    "Ansor (unfused)": unfused_list,
    #    "Ansor (fused)": tvm_fused_list,
    #    "Souffle (reduce kernel)": one_kernel_list,
    #    "Souffle (data reuse)": fused_list
    # }
    fusion_latency = {
       "unfused": unfused_list,
       "fused": tvm_fused_list,
       "global-sync": one_kernel_list,
       "data-reuse": fused_list
    }
    print(fusion_latency)
    draw_bars(species, fusion_latency, "efficientnet-se-module-latency-ours.pdf", 3)
    draw_bars(species, fusion_latency, "efficientnet-se-module-latency-ours.svg", 3)

    return unfused_list, tvm_fused_list, one_kernel_list, fused_list


def draw_efficientnet_plot_v2(group_data):
    file_path = "/home/xiachunwei/Software/tensor-compiler/src/operator_fusion/models/efficientnet/tvm-swish-module-latency.npy"
    blocks_data = np.load(file_path, allow_pickle=True)
    unfused_list = blocks_data[0]
    tvm_fused_list = blocks_data[1]
    one_kernel_list, fused_list = [], []
    for group in group_data:
        group = np.array(group)
        one_kernel = group[9]
        fused = group[11]
        one_kernel_list.append(one_kernel)
        fused_list.append(fused)
        print("{} {}".format(one_kernel, fused))
    species = []
    for i in range(len(group_data)):
       species.append("M{}".format(i))
    fusion_latency = {
       "Ansor (unfused)": np.array(unfused_list),
       "Ansor (fused)": np.array(tvm_fused_list),
       "Souffle (reduce kernel)": np.array(one_kernel_list),
       "Souffle (data reuse)": np.array(fused_list)
    }
    print(fusion_latency)
    draw_bars(species, fusion_latency, "efficientnet-se-module-latency-souffle.pdf", 100)

    print(total_latency(unfused_list), total_latency(tvm_fused_list), total_latency(one_kernel_list), total_latency(fused_list))
    return unfused_list, tvm_fused_list, one_kernel_list, fused_list


if __name__=="__main__":
    # data_groups = read_ncu_xls("./efficientnet-se_module_v2-max-block.xlsx")
    # print(data_groups)
    data_groups = read_ncu_csv("../ncu-efficient_se_module_unittest-kernel-latency.csv")
    draw_efficientnet_plot(data_groups, normalize=True)
    # draw_efficientnet_plot_v2(data_groups)

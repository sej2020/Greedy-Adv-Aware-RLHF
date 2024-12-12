import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

### Global Style Parameters ###
##############################
SMALL_SIZE = 10
MEDIUM_SIZE = 14
BIGGER_SIZE = 18
CHONK_SIZE = 24
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('axes', titlesize=BIGGER_SIZE, labelsize=MEDIUM_SIZE, facecolor="xkcd:red")
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=CHONK_SIZE, facecolor="xkcd:white", edgecolor="xkcd:red") 
sns.set_style("darkgrid", {'font.family':['serif'], 'axes.edgecolor':'black','ytick.left': True})
plt.ticklabel_format(style = 'plain')
# plt.rcParams['axes.edgecolor'] = 'black'
###############################

def main():
    exp_data = json.load(open("plotting/sharpness/exp_data.json", "r"))
    unexp_data = json.load(open("plotting/sharpness/unexp_data.json", "r"))
    for metric in ["loss_list_full", "loss_list_unembed"]:

        fig, ax = plt.subplots()
        for idx, (key, value) in enumerate(exp_data.items()):
            if value[metric][0] > value[metric][-1]:
                y = value[metric][::-1]
            else:
                y = value[metric]
            ax.plot(np.linspace(-1.0, 1.0, 41).astype(np.float32), y)
        fixed_axis = ax.get_ylim()
        ax.set_title(f"Parameter Landscape")
        ax.set_ylabel("Objective Value")
        ax.set_xlabel("Parameter Perturbation (Last Layer Only)" if "unembed" in metric else "Parameter Perturbation")
        fig.savefig(f"plotting/sharpness/exp_{metric.split('_')[-1]}.png")

        fig, ax = plt.subplots()
        for idx, (key, value) in enumerate(unexp_data.items()):
            if value[metric][0] > value[metric][-1]:
                y = value[metric][::-1]
            else:
                y = value[metric]
            ax.plot(np.linspace(-1.0, 1.0, 41).astype(np.float32), y)
        ax.set_ylim(fixed_axis)
        ax.set_title(f"Parameter Landscape")
        ax.set_ylabel("Objective Value")
        ax.set_xlabel("Parameter Perturbation (Last Layer Only)" if "unembed" in metric else "Parameter Perturbation")
        fig.savefig(f"plotting/sharpness/unexp_{metric.split('_')[-1]}.png")



if __name__ == '__main__':
    main()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

def main(metric, project):

    if project == 'Expr1_Philo_Capped':
        ent_coef = 0.001
        kl_coef = 1.5
    elif project == 'Expr1_Philo_Uncapped':
        ent_coef = 0.001
        kl_coef = 1.0

    match metric:
        case 'kl':
            ytick_max = None
            title = 'KL Divergence with Exploitable RM'
            ylabel = 'KL Divergence'
            if project == 'Expr1_Philo_Capped':
                coef = 1.5
            elif project == 'Expr1_Philo_Uncapped':
                coef = 1.0
        case 'ent':
            ytick_max = None
            title = 'Entropy with Exploitable RM'
            ylabel = 'Entropy'
            coef = 0.001
        case 'reward':
            ytick_min = 0
            if project == 'Expr1_Philo_Capped':
                ytick_max = 1.1
                ytick_step = 0.1
            elif project == 'Expr1_Philo_Uncapped':
                ytick_max = 2.1
                ytick_step = 0.25
            title = 'Training Reward with Exploitable RM'
            ylabel = 'Reward'
            coef = 1.0
        case 'eval':
            return eval_ana(project)
        case _:
            print('Invalid metric')
            raise ValueError

    gaa_df = pd.read_csv(f'plotting/expr1_result_csv/{project.split('_')[-1]}/{metric}_{project}_gaa.csv', index_col=0) / coef
    gaa_mean = gaa_df.apply(np.mean, axis=1)
    gaa_std = gaa_df.apply(np.std, axis=1)

    gaa_lower_ci = gaa_mean - 1.96 * gaa_std / np.sqrt(gaa_df.shape[1])
    gaa_upper_ci = gaa_mean + 1.96 * gaa_std / np.sqrt(gaa_df.shape[1])
    plt.plot(gaa_mean, label='GAA', color='blueviolet')
    plt.fill_between(gaa_mean.index, gaa_lower_ci, gaa_upper_ci, alpha=0.2, color='blueviolet')

    baseline_df = pd.read_csv(f'plotting/expr1_result_csv/{project.split('_')[-1]}/{metric}_{project}_baseline.csv', index_col=0) / coef
    baseline_mean = baseline_df.apply(np.mean, axis=1)
    baseline_std = baseline_df.apply(np.std, axis=1)

    baseline_lower_ci = baseline_mean - 1.96 * baseline_std / np.sqrt(baseline_df.shape[1])
    baseline_upper_ci = baseline_mean + 1.96 * baseline_std / np.sqrt(baseline_df.shape[1])

    plt.plot(baseline_mean, label='Baseline', color='darkslategrey')
    plt.fill_between(baseline_mean.index, baseline_lower_ci, baseline_upper_ci, alpha=0.2, color='darkslategrey')

    plt.legend()
    plt.xlabel('Epoch')
    if ytick_max:
        plt.yticks(np.arange(ytick_min, ytick_max, ytick_step))
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(f'plotting/expr1_result_plots/{project.split('_')[-1]}/{metric}_{project}.png')
    plt.clf()

def eval_ana(project):
    gaa_df = pd.read_csv(f'plotting/expr1_result_csv/{project.split('_')[-1]}/eval_reward_{project}_gaa.csv', index_col=0)
    gaa_mean = gaa_df.apply(np.mean, axis=1)
    gaa_std = gaa_df.apply(np.std, axis=1)

    baseline_df = pd.read_csv(f'plotting/expr1_result_csv/{project.split('_')[-1]}/eval_reward_{project}_baseline.csv', index_col=0)
    baseline_mean = baseline_df.apply(np.mean, axis=1)
    baseline_std = baseline_df.apply(np.std, axis=1)

    print(f"GAA: {gaa_mean.iloc[-1]:.4f} +/- {1.96 * gaa_std.iloc[-1] / np.sqrt(gaa_df.shape[1]):.4f}")
    print(f"Baseline: {baseline_mean.iloc[-1]:.4f} +/- {1.96 * baseline_std.iloc[-1] / np.sqrt(baseline_df.shape[1]):.4f}")


if __name__ == '__main__':
    # metric = 'reward' # 'kl', 'ent', 'reward', 'eval'
    # project = 'Expr1_Philo_Capped'
    for metric in ['kl', 'ent', 'reward']:
        for project in ['Expr1_Philo_Capped', 'Expr1_Philo_Uncapped']:
            main(metric, project)
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from collections import OrderedDict

from glob import glob

from io_funcs import load_config

#######################################################################################################

PROJ = 'direct_training_gru_relobralo'

OUT_DIR = os.path.join('outputs', PROJ)

MECH_TEST = 'x15_y15_'

EXCLUTIONS = ['confused-grass-51', 'proud-haze-52', 'floral-sun-53']

models = glob(os.path.join(OUT_DIR,'models/*'))
stats = glob(os.path.join(OUT_DIR,'val/*'))

runs = dict.fromkeys([model.split('\\')[-1] for model in models if model.split('\\')[-1] not in EXCLUTIONS])
runs_2 = OrderedDict.fromkeys([model.split('\\')[-1] for model in models if model.split('\\')[-1] not in EXCLUTIONS])

for k,_ in runs.items():

    if k not in EXCLUTIONS:
        model_path = next(x for x in models if k in x)
        stats_path = next(x for x in stats if k in x)

        config = load_config(model_path, 'config.yaml')
        model_stats = pd.read_csv(os.path.join(stats_path,'stats.csv'))
        elem_stats = pd.read_csv(os.path.join(stats_path, 'stats_elems.csv'))

        params = [config.train.loss_settings.alpha,
                config.train.loss_settings.tau,
                model_stats['RMSE'].median(),
                elem_stats['rmse_s0'][0],
                model_stats[model_stats['Run']==MECH_TEST].values[0,1]]

        runs[k] = params
        runs_2[k] = model_stats['RMSE'].values

data = pd.DataFrame(list(runs.values())).sort_values([0,1])
alldata = np.column_stack(list(runs_2.values()))

alpha = sorted(set(data[0]))
tau = sorted(set(data[1]), reverse=True)

mats = {i: np.zeros((len(tau),len(alpha))) for i in range(len(params)-2)}

labels = {0: 'Median RMSE Validation Set',
          1: r'RMSE $\sigma_0$',
          2: f'Median RMSE {MECH_TEST}'}

for i, a in enumerate(alpha):
    for j, t in enumerate(tau):
        for k in mats.keys():
            mats[k][j][i] = data.loc[(data[0]==a) & (data[1]==t), k+2].item()

fig, axs = plt.subplots(1,3)
fig.set_size_inches(9.2,5.4)
fig.tight_layout()
fig.subplots_adjust(bottom=0.15, wspace=0.25, left=0.055, right=0.955)


for i, ax in enumerate(axs.flatten()):
    min = np.min(mats[i])
    max = np.max(mats[i])
    p=sns.heatmap(mats[i], annot=True, fmt=".4f", xticklabels=alpha, yticklabels=tau, square=True, ax=ax, cbar_kws={"shrink": .78,'orientation': 'horizontal', 'ticks': np.linspace(min, max, 4)}, vmin=min,  vmax=max)
    p.set(xlabel=r'$\alpha$', ylabel=r'$\mathcal{T}$')
    cbar = p.collections[0].colorbar 
    cbar.set_label(labels[i], labelpad=12.5)

plt.savefig(os.path.join(OUT_DIR, 'corr_runs.png'), format="png", dpi=600, bbox_inches='tight')
plt.clf()
plt.close(fig)

fig, ax = plt.subplots(1,1)
fig.set_size_inches(9.2,6.2)


plt.boxplot(alldata, patch_artist=True, labels=runs_2.keys())
plt.xticks(rotation=45, ha='right')
plt.ylabel('RMSE validation set')
fig.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'box_plot.png'), format="png", dpi=600, bbox_inches='tight')
plt.clf()
plt.close(fig)




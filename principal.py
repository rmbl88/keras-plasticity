import numpy as np
import pandas as pd

from functions import data_sampling, load_dataframes
from constants import *
import itertools
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.colors import to_rgba

# Loading data
df_list, _ = load_dataframes(TRAIN_MULTI_DIR)

# Sampling data pass random seed for random sampling
sampled_dfs = data_sampling(df_list, DATA_SAMPLES)

# Merging training data
data = pd.concat(sampled_dfs, axis=0, ignore_index=True)

# Reorganizing dataset by time increment, subsequent grouping by tag and final shuffling
data_by_t = [df for _, df in data.groupby(['t'])]
data_by_tag = [[df for _, df in group.groupby(['tag'])] for group in data_by_t]
data_by_batches = list(itertools.chain(*data_by_tag))

#Concatenating data groups
data = pd.concat(data_by_batches).reset_index(drop=True)

tags = list(set(data['tag'].values.tolist()))

for tag in tags:

    df = data[data['tag']==tag]
    condition = df[df['t']==1.0]

    s_xx = condition['sxx_t'].values
    s_yy = condition['syy_t'].values
    s_xy = condition['sxy_t'].values

    e_xx = condition['exx_t'].values
    e_yy = condition['eyy_t'].values
    e_xy = condition['exy_t'].values

    s1 = 0.5 * (s_xx + s_yy) + ((0.5 * (s_xx-s_yy))**2 + s_xy**2)**0.5
    s2 = 0.5 * (s_xx + s_yy) - ((0.5 * (s_xx-s_yy))**2 + s_xy**2)**0.5

    e_1 = 0.5 * (e_xx + e_yy) + ((0.5 * (e_xx-e_yy))**2 + e_xy**2)**0.5
    e_2 = 0.5 * (e_xx + e_yy) - ((0.5 * (e_xx-e_yy))**2 + e_xy**2)**0.5

    w = 2*160*2**0.5
    h = 2*160*(2/3)**0.5

    el = Ellipse((0,0),w,h,45,ls='-', ec='0.25', fc=to_rgba('0.35', 0.10))

    plt.rcParams.update(PARAMS)
    fig, (ax1,ax2) = plt.subplots(1,2, figsize=(12,5))
    fig.suptitle(tag.replace("_","\_"))

    bbox={'alpha':0.65, 'facecolor':'white', 'edgecolor':'None'}

    ax1.grid(linestyle='--')
    ax1.add_patch(el)
    ax1.scatter(s2,s1,marker='o', facecolors='white', edgecolors='r',zorder=10)
    ax1.set_xlabel(r'$\sigma_2$')
    ax1.set_ylabel(r'$\sigma_1$')
    xabs_max_0 = abs(max(ax1.get_xlim(), key=abs))
    ax1.set_xlim(xmin=-xabs_max_0, xmax=xabs_max_0)
    ax1.set_ylim(ymin=0.0, ymax=max(ax1.get_ylim())*1.1)
    

    #axs[0].set_aspect(abs((x_right-x_left)/(y_low-y_high))*1.0)

    ax1.set_aspect(1.0/ax1.get_data_ratio(), adjustable='box')
 


    l1 = ax1.plot([0,0], [0,max(ax1.get_ylim())], "k--", linewidth=1)
    ax1.text(max(ax1.get_xlim())*0.2, max(ax1.get_ylim())*0.88, 'Uniaxial\ntension', ha='center', bbox=bbox)
    l2 = ax1.plot([0,min(ax1.get_xlim())],[0, max(ax1.get_xlim())], "k--", linewidth=1)
    ax1.text(min(ax1.get_xlim())*0.75, max(ax1.get_xlim()), 'Shear', ha='center', bbox=bbox)
    l3 = ax1.plot([0,max(ax1.get_xlim())],[0, max(ax1.get_xlim())], "k--", linewidth=1)
    ax1.text(max(ax1.get_xlim())*0.68, max(ax1.get_xlim()),'Equi-biaxial\ntension', ha='center', bbox=bbox)



    ax2.grid(linestyle='--')
    ax2.scatter(e_2,e_1,marker='o', facecolors='white', edgecolors='g', zorder=10)
    ax2.set_xlabel(r'$\varepsilon_2$')
    ax2.set_ylabel(r'$\varepsilon_1$')
    xabs_max_1 = abs(max(ax2.get_xlim(), key=abs))
    ax2.set_xlim(xmin=-xabs_max_1, xmax=xabs_max_1)

    x_left, x_right = ax2.get_xlim()
    y_low, y_high = ax2.get_ylim()
    #axs[1].set_aspect(abs((x_right-x_left)/(y_low-y_high))*1.0)

    ax1.set_aspect(1.0/ax1.get_data_ratio(), adjustable='box')


    l4 = ax2.plot([0,min(ax2.get_xlim())],[0, max(ax2.get_xlim())], "k--", linewidth=1)
    ax2.text(min(ax2.get_xlim())*0.75, max(ax2.get_xlim()), 'Shear', ha='center', bbox=bbox)
    l5 = ax2.plot([0,max(ax2.get_xlim())],[0, max(ax2.get_xlim())], "k--", linewidth=1)
    ax2.text(max(ax2.get_xlim())*0.68, max(ax2.get_xlim()), 'Equi-biaxial\ntension', ha='center', bbox=bbox)
    l6 = ax2.plot([0,-max(ax2.get_ylim())*0.5],[0, max(ax2.get_ylim())], "k--", linewidth=1)
    ax2.text(min(ax2.get_xlim())*0.4, max(ax2.get_ylim())*0.7, 'Uniaxial tension\n(isotropy)', ha='center', bbox=bbox)



    #plt.show()
    plt.savefig('principal_%s.png' % (tag), format='png',dpi=300, bbox_inches='tight')
    plt.close('all')

print('hey')
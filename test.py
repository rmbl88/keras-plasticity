import numpy as np
import matplotlib.pyplot as plt
from constants import *
from mpl_toolkits import mplot3d
import math
import os
import pandas as pd
from cycler import cycler
from matplotlib.ticker import MaxNLocator

def set_size(width, fraction=1, subplots=(1, 1)):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == 'thesis':
        width_pt = 426.79135
    elif width == 'beamer':
        width_pt = 307.28987
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1]) * 1.5

    return (fig_width_in, fig_height_in)

file_list = []
df_list = []

DIR = "outputs/9-elem-200-plastic_testfull/val/"

for r, d, f in os.walk(DIR):
    for file in sorted(f):
        if 'm12_b200_9.csv' in file:
            file_list.append(DIR + file)

for file in file_list:
    df_list.append(pd.read_csv(file))

plt.rcParams.update(PARAMS)

default_cycler = (cycler(color=["#ef476f","#118ab2","#073b4c"]))

plt.rc('axes', prop_cycle=default_cycler)



# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

ex_var_abaqus = df_list[0]['exx_t']*100
ey_var_abaqus = df_list[0]['eyy_t']*100
exy_var_abaqus = df_list[0]['exy_t']*100
sx_var_abaqus = df_list[0]['sxx_t']
sy_var_abaqus = df_list[0]['syy_t']
sxy_var_abaqus = df_list[0]['sxy_t']

sx_pred_var = df_list[0]['pred_x']
sy_pred_var = df_list[0]['pred_y']
sxy_pred_var = df_list[0]['pred_xy']

fig , (ax1, ax2, ax3) = plt.subplots(1,3)
# fig.set_size_inches(12, 3.35)
fig.set_size_inches(set_size(484,subplots=(1, 3)))
fig.subplots_adjust(bottom=0.28, wspace=0.4)

ax1.plot(ex_var_abaqus, sx_var_abaqus, label='ABAQUS',marker='o',linestyle='None',alpha=.2,markersize=2.25, color='k')
ax1.plot(ex_var_abaqus, sx_pred_var, label='ANN',marker='o',linestyle='None',alpha=.1,markersize=2.25,color='r')
ax1.set(xlabel=r'$\varepsilon_{xx}$ [\%]', ylabel=r'$\sigma_{xx}$ [MPa]')
#ax1.set_title(r'$\text{MSE}=%0.3f$' % (mse_x), fontsize=11)
ax2.plot(ey_var_abaqus, sy_var_abaqus, label='ABAQUS',marker='o',linestyle='None',alpha=.2,markersize=2.25, color='k')
ax2.plot(ey_var_abaqus, sy_pred_var, label='ANN',marker='o',linestyle='None',alpha=.1,markersize=2.25,color='r')
ax2.set(xlabel=r'$\varepsilon_{yy}$ [\%]', ylabel=r'$\sigma_{yy}$ [MPa]')
#ax2.set_title(r'$\text{MSE}=%0.3f$' % (mse_y), fontsize=11)
ax3.plot(exy_var_abaqus, sxy_var_abaqus, label='ABAQUS',marker='o',linestyle='None',alpha=.2,markersize=2.25, color='k')
ax3.plot(exy_var_abaqus, sxy_pred_var, label='ANN',marker='o',linestyle='None',alpha=.1,markersize=2.25,color='r')
ax3.set(xlabel=r'$\varepsilon_{xy}$ [\%]', ylabel=r'$\tau_{xy}$ [MPa]')
#ax3.set_title(r'$\text{MSE}=%0.3f$' % (mse_xy), fontsize=11)
handles, labels = ax3.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center',ncol=2)

ax1.tick_params(axis='x', labelcolor='black', length=6)
ax1.tick_params(axis='y', labelcolor='black',length=6)
ax2.tick_params(axis='x', labelcolor='black', length=6)
ax2.tick_params(axis='y', labelcolor='black', length=6)
ax2.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
ax3.tick_params(axis='x', labelcolor='black', length=6)
ax3.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
ax3.tick_params(axis='y', labelcolor='black', length=6)

ax1.xaxis.set_major_locator(MaxNLocator(6))
ax1.yaxis.set_major_locator(MaxNLocator(6)) 

ax2.xaxis.set_major_locator(MaxNLocator(6))
ax2.yaxis.set_major_locator(MaxNLocator(6)) 

ax3.xaxis.set_major_locator(MaxNLocator(6))
ax3.yaxis.set_major_locator(MaxNLocator(6)) 


#plt.show()
plt.savefig("outputs/9-elem-200-plastic_testfull/val/val_m12_b200_9.pdf", format="pdf", dpi=300, bbox_inches='tight')
print('hey')
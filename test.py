import numpy as np
import matplotlib.pyplot as plt
import torchmetrics
from constants import *
from mpl_toolkits import mplot3d
import math
import os
import pandas as pd
from cycler import cycler
from matplotlib.ticker import MaxNLocator
import torch
from matplotlib.offsetbox import AnchoredText

def set_anchored_text(mse,r2,mse_d,r2_d,frameon=True,loc='upper left'):
    at = AnchoredText('MAE: %.5f\n$r^{2}$: %.5f\nMAE(D): %.5f\n$r^{2}$(D): %.5f' % (mse,r2,mse_d,r2_d), loc=loc, frameon=frameon,prop=dict(fontsize=7))
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    return at

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
    elif width == 'esaform':
        width_pt = 535.896
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

DIR = "outputs/crux-plastic_sbvf_abs_direct/val/"

for r, d, f in os.walk(DIR):
    for file in sorted(f):
        if '.csv' in file:
            file_list.append(DIR + file)

for file in file_list:
    df_list.append(pd.read_csv(file))

plt.rcParams.update(PARAMS_CONTOUR)

default_cycler = (cycler(color=["#ef476f","#118ab2","#073b4c"]))

plt.rc('axes', prop_cycle=default_cycler)
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')

err = torchmetrics.MeanAbsoluteError()
r2 = torchmetrics.R2Score()

for i,file in enumerate(file_list):

    str = file.split('/')[-1][:-4]
    if '_x_' in str:
        expr = '_x_'
    elif '__' in str:
        expr = '__'
    else:
        expr = '_y_'
    
    trial = str.split(expr)[0]
    elem = str.split(expr)[-1]

    ex_var_abaqus = df_list[i]['e_xx']
    ey_var_abaqus = df_list[i]['e_yy']
    exy_var_abaqus = df_list[i]['e_xy']
    sx_var_abaqus = df_list[i]['s_xx']
    sy_var_abaqus = df_list[i]['s_yy']
    sxy_var_abaqus = df_list[i]['s_xy']

    e_1_abaqus = df_list[i]['e_1']
    e_2_abaqus = df_list[i]['e_2']

    s_1_abaqus = df_list[i]['s_1']
    s_2_abaqus = df_list[i]['s_2']

    s_1_pred = df_list[i]['s_1_pred']
    s_2_pred = df_list[i]['s_2_pred']

    sx_pred_var = df_list[i]['s_xx_pred']
    sy_pred_var = df_list[i]['s_yy_pred']
    sxy_pred_var = df_list[i]['s_xy_pred']

    de_1_abaqus = df_list[i]['de_1']
    de_2_abaqus = df_list[i]['de_2']

    dy_1_abaqus = df_list[i]['dy_1']
    dy_2_abaqus = df_list[i]['dy_2']

    ds_1_pred = df_list[i]['ds_1']
    ds_2_pred = df_list[i]['ds_2']

    fig , axs = plt.subplots(1,3)
    #fig.set_size_inches(16, 9)
    fig.set_size_inches(set_size('esaform',subplots=(1, 3)))
    fig.subplots_adjust(bottom=0.28, wspace=0.35)
    fig.suptitle(r'' + trial.replace('_','\_') + ': element \#' + elem, fontsize=9)

    axs[0].plot(ex_var_abaqus, sx_var_abaqus, label='ABAQUS', color='k', marker='.', markersize=0.75)
    axs[0].plot(ex_var_abaqus, sx_pred_var, label='ANN', color='r', marker='.', markersize=0.9, alpha=0.65)
    #axs[0].plot(ex_var_abaqus, sx_pred_var_2, label='ANN(D)',color='b')
    axs[0].set(xlabel=r'$\varepsilon_{xx}$', ylabel=r'$\sigma_{xx}$ [MPa]')
    #axs[0].add_artist(set_anchored_text(mse_x,r2_x,mse_x_2,r2_x_2))
    #axs[0].set_title(r'$\text{MSE}=%0.3f$' % (mse_x), fontsize=11)
    axs[1].plot(ey_var_abaqus, sy_var_abaqus, label='ABAQUS', color='k', marker='.', markersize=0.75)
    axs[1].plot(ey_var_abaqus, sy_pred_var, label='ANN', color='r', marker='.', markersize=0.9, alpha=0.65)
    #axs[1].plot(ey_var_abaqus, sy_pred_var_2, label='ANN(D)',color='b')
    axs[1].set(xlabel=r'$\varepsilon_{yy}$', ylabel=r'$\sigma_{yy}$ [MPa]')
    #axs[1].add_artist(set_anchored_text(mse_y,r2_y,mse_y_2,r2_y_2))
    #axs[1].set_title(r'$\text{MSE}=%0.3f$' % (mse_y), fontsize=11)
    axs[2].plot(exy_var_abaqus, sxy_var_abaqus, label='ABAQUS', color='k', marker='.', markersize=0.75)
    axs[2].plot(exy_var_abaqus, sxy_pred_var, label='ANN', color='r', marker='.', markersize=0.9, alpha=0.65)
    #axs[2].plot(exy_var_abaqus, sxy_pred_var_2, label='ANN(D)',color='b')
    axs[2].set(xlabel=r'$\varepsilon_{xy}$', ylabel=r'$\tau_{xy}$ [MPa]')
    #axs[2].add_artist(set_anchored_text(mse_xy,r2_xy,mse_xy_2,r2_xy_2))
    #axs[2].set_title(r'$\text{MSE}=%0.3f$' % (mse_xy), fontsize=11)
    
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center',ncol=2)

    for ax in fig.axes:

        ax.tick_params(axis='x', labelcolor='black', length=6)
        ax.tick_params(axis='y', labelcolor='black',length=6)
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    
        ax.xaxis.set_major_locator(MaxNLocator(6))
        ax.yaxis.set_major_locator(MaxNLocator(6)) 

    #plt.show()
    plt.savefig(DIR + trial + expr + elem + "_1.png", format="png", dpi=600, bbox_inches='tight')
    plt.close(fig)

    print('saving %i'% i)

    fig , axs = plt.subplots(3,2)
    #fig.set_size_inches(16, 9)
    fig.set_size_inches(set_size('esaform',subplots=(2, 2)))
    fig.subplots_adjust(bottom=0.1, wspace=0.35)
    fig.suptitle(r'' + trial.replace('_','\_') + ': element \#' + elem, fontsize=9)

    axs[0][0].plot(e_1_abaqus, s_1_abaqus, label='ABAQUS', color='k',  marker='.',markersize=2.2, alpha=0.65)
    axs[0][0].plot(e_1_abaqus, s_1_pred, label='ANN',color='r',  marker='.',markersize=2.2, alpha=0.5)
    axs[0][0].set(xlabel=r'$\varepsilon_{1}$', ylabel=r'$\sigma_{1}$ [MPa]')

    axs[0][1].plot(e_2_abaqus, s_2_abaqus, label='ABAQUS', color='k',  marker='.',markersize=2.2, alpha=0.65)
    axs[0][1].plot(e_2_abaqus, s_2_pred, label='ANN',color='r',marker='.',markersize=2.2, alpha=0.5)
    axs[0][1].set(xlabel=r'$\varepsilon_{2}$', ylabel=r'$\sigma_{2}$ [MPa]')

    axs[1][0].plot(dy_1_abaqus, label='ABAQUS', color='k',  marker='.',markersize=2.2, alpha=0.65)
    axs[1][0].plot(ds_1_pred, label='ANN',color='r',  marker='.',markersize=2.2, alpha=0.5)
    axs[1][0].set(xlabel=r'$t$', ylabel=r'$\dot{\sigma}_{1}$ [MPa/s]')

    axs[1][1].plot(dy_2_abaqus, label='ABAQUS', color='k',  marker='.',markersize=2.2, alpha=0.65)
    axs[1][1].plot(ds_2_pred, label='ANN',color='r',  marker='.',markersize=2.2, alpha=0.5)
    axs[1][1].set(xlabel=r'$t$', ylabel=r'$\dot{\sigma}_{2}$ [MPa/s]')

    axs[2][0].plot(dy_1_abaqus*de_1_abaqus, label='ABAQUS', color='k',  marker='.',markersize=2.2, alpha=0.65)
    axs[2][0].plot(ds_1_pred*de_1_abaqus, label='ANN',color='r',  marker='.',markersize=2.2, alpha=0.5)
    axs[2][0].set(xlabel=r'$t$', ylabel=r'$\dot{\sigma}_{1}\dot{\varepsilon}_{1}$')

    axs[2][1].plot(dy_2_abaqus*de_2_abaqus, label='ABAQUS', color='k',  marker='.',markersize=2.2, alpha=0.65)
    axs[2][1].plot(ds_2_pred*de_2_abaqus, label='ANN',color='r',  marker='.',markersize=2.2, alpha=0.5)
    axs[2][1].set(xlabel=r'$t$', ylabel=r'$\dot{\sigma}_{2}\dot{\varepsilon}_{2}$')

    handles, labels = axs[0][1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center',ncol=2)

    for ax in fig.axes:

        ax.tick_params(axis='x', labelcolor='black', length=6)
        ax.tick_params(axis='y', labelcolor='black',length=6)
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    
        ax.xaxis.set_major_locator(MaxNLocator(6))
        ax.yaxis.set_major_locator(MaxNLocator(6)) 
    
    #plt.show()
    plt.savefig(DIR + trial + expr + elem + "_2.png", format="png", dpi=600, bbox_inches='tight')
    plt.close(fig)

    print('\rProcessing image %i of %i' % (i+1,len(file_list)), end='')

from dask import get
import matplotlib
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
import glob
from tqdm import tqdm

def get_stats(var):
    return (np.mean(var),np.median(var),max(var),min(var))

def set_anchored_text(mean_e,median_e,max_e,min_e,frameon=True,loc='upper right'):
    at = AnchoredText(f'Mean: {np.round(mean_e,3)}\nMedian: {np.round(median_e,3)}\nMax.: {np.round(max_e,3)}\nMin.: {np.round(min_e,3)}', loc=loc, frameon=frameon,prop=dict(fontsize=5.5))
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    at.patch.set_linewidth(0.55)
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


df_list = []

RUN = 'woven-field-208'
#RUN = 'whole-puddle-134'
#DIR = f'outputs/crux-plastic_sbvf_abs_direct/val/{RUN}/'
DIR = f'outputs/sbvfm_indirect_crux_gru/val/{RUN}/'

mech_tests = []
for r, d, f in os.walk(DIR):
    for folder in d:
        mech_tests.append(folder)

mech_tests = {k: None for k in mech_tests}

for k_test,v in mech_tests.items():
    
    TEST_PATH = os.path.join(DIR, k_test)

    file_list = glob.glob(os.path.join(TEST_PATH, 'data', f'*.csv'))

    mech_tests[k_test] = {file.split('-')[-1][:-4]:pd.read_csv(file) for file in file_list}

matplotlib.use('Agg')
plt.rcParams.update(PARAMS)

default_cycler = (cycler(color=["#ef476f","#118ab2","#073b4c"]))

plt.rc('axes', prop_cycle=default_cycler)
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')

err = torchmetrics.MeanAbsoluteError()
r2 = torchmetrics.R2Score()

TAG = 'x15_y15_'

for k_test, elems in (pbar_1 := tqdm(mech_tests.items(), bar_format=FORMAT_PBAR, leave=False)):
    
    pbar_1.set_description(f'Processing trial - {k_test}')

    if k_test == TAG:
    
        for elem, df in (pbar_2 := tqdm(elems.items(), bar_format=FORMAT_PBAR, leave=False)):
            
            pbar_2.set_description(f'Plotting elem-{elem}')

            TRIAL = k_test
            ELEM = elem

            TITLE = r'' + TRIAL.replace('_','\_') + ' - element \#' + ELEM
            
            ex_var_abaqus = df['e_xx'].values
            ey_var_abaqus = df['e_yy'].values
            exy_var_abaqus = df['e_xy'].values
            sx_var_abaqus = df['s_xx'].values
            sy_var_abaqus = df['s_yy'].values
            sxy_var_abaqus = df['s_xy'].values

            # e_1_abaqus = df['e_1']
            # e_2_abaqus = df['e_2']

            # s_1_abaqus = df['s_1']
            # s_2_abaqus = df['s_2']

            # s_1_pred = df['s_1_pred']
            # s_2_pred = df['s_2_pred']

            sx_pred_var = df['s_xx_pred'].values
            sy_pred_var = df['s_yy_pred'].values
            sxy_pred_var = df['s_xy_pred'].values

            err_x = np.abs(sx_var_abaqus - sx_pred_var)
            
            err_y = np.abs(sy_var_abaqus - sy_pred_var)
            
            err_xy = np.abs(sxy_var_abaqus - sxy_pred_var)
        

            # de_1_abaqus = df['de_1']
            # de_2_abaqus = df['de_2']

            # dy_1_abaqus = df['dy_1']
            # dy_2_abaqus = df['dy_2']

            # ds_1_pred = df['ds_1']
            # ds_2_pred = df['ds_2']

            fig , axs = plt.subplots(1,3)
            #fig.set_size_inches(16, 9)
            fig.set_size_inches(set_size('esaform',subplots=(1, 3)))
            fig.subplots_adjust(bottom=0.28, wspace=0.35)
            fig.suptitle(TITLE, fontsize=9)
            
            axs[0].plot(ex_var_abaqus, sx_var_abaqus, label='ABAQUS', color='k', marker='.', markersize=0.75)
            axs[0].plot(ex_var_abaqus, sx_pred_var, label='ANN', color='r', marker='.', markersize=0.9, alpha=0.65)
            axs[0].set(xlabel=r'$\varepsilon_{xx}$', ylabel=r'$\sigma_{xx}$ [MPa]')

            axs[1].plot(ey_var_abaqus, sy_var_abaqus, label='ABAQUS', color='k', marker='.', markersize=0.75)
            axs[1].plot(ey_var_abaqus, sy_pred_var, label='ANN', color='r', marker='.', markersize=0.9, alpha=0.65)
            axs[1].set(xlabel=r'$\varepsilon_{yy}$', ylabel=r'$\sigma_{yy}$ [MPa]')

            axs[2].plot(exy_var_abaqus, sxy_var_abaqus, label='ABAQUS', color='k', marker='.', markersize=0.75)
            axs[2].plot(exy_var_abaqus, sxy_pred_var, label='ANN', color='r', marker='.', markersize=0.9, alpha=0.65)
            axs[2].set(xlabel=r'$\varepsilon_{xy}$', ylabel=r'$\tau_{xy}$ [MPa]')

            handles, labels = axs[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower center',ncol=2)

            for ax in fig.axes:

                ax.tick_params(axis='x', labelcolor='black', length=6)
                ax.tick_params(axis='y', labelcolor='black',length=6)
                ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
            
                ax.xaxis.set_major_locator(MaxNLocator(6))
                ax.yaxis.set_major_locator(MaxNLocator(6)) 
        
            plt.savefig(os.path.join(DIR,k_test,'plots', f'{TRIAL}_el-{ELEM}_a.png'), format="png", dpi=600, bbox_inches='tight')
            plt.clf()
            plt.close(fig)
            
            fig , axs = plt.subplots(1,3)
            #fig.set_size_inches(16, 9)
            fig.set_size_inches(set_size('esaform',subplots=(1, 3)))
            fig.subplots_adjust(bottom=0.28, wspace=0.3)
            fig.suptitle(TITLE, fontsize=9)
            
            colors = ['lightgray'] * err_x.shape[0]

            # axs[0].bar(df.index.values, err_x, lw=0.2, width=0.8, ec="white", fc="black", alpha=0.5, align='edge')
            # axs[1].bar(df.index.values, err_y, lw=0.2, width=0.8, ec="white", fc="black", alpha=0.5, align='edge')
            # axs[2].bar(df.index.values, err_xy, lw=0.2, width=0.8, ec="white", fc="black", alpha=0.5, align='edge')

            vars_lbl = [r'$\sigma_{xx}$', r'$\sigma_{yy}$', r'$\tau_{xy}$']
            vars = [err_x, err_y, err_xy]

            for i, ax in enumerate(fig.axes):
                axs[i].bar(df.index.values, vars[i], lw=0.1, width=0.65, ec="white", fc="black", alpha=0.5, align='center')
                ax.set(xlabel=r'Time steps', ylabel=r'%s - Abs. error [MPa]' % (vars_lbl[i]))
                axs[i].add_artist(set_anchored_text(*get_stats(vars[i])))
                # Choose the interval to space out xticks
                _min = np.min(df.index.values)
                diff = 20
                # Decide positions of ticks
                ticks = np.arange(_min, np.max(df.index.values), diff)
                # Also, choose labels of the ticks
                axs[i].set_xticks(ticks=ticks)
                
                
            # axs[0][0].plot(e_1_abaqus, s_1_abaqus, label='ABAQUS', color='k',  marker='.',markersize=2.2, alpha=0.65)
            # axs[0][0].plot(e_1_abaqus, s_1_pred, label='ANN',color='r',  marker='.',markersize=2.2, alpha=0.5)
            # axs[0][0].set(xlabel=r'$\varepsilon_{1}$', ylabel=r'$\sigma_{1}$ [MPa]')

            # axs[0][1].plot(e_2_abaqus, s_2_abaqus, label='ABAQUS', color='k',  marker='.',markersize=2.2, alpha=0.65)
            # axs[0][1].plot(e_2_abaqus, s_2_pred, label='ANN',color='r',marker='.',markersize=2.2, alpha=0.5)
            # axs[0][1].set(xlabel=r'$\varepsilon_{2}$', ylabel=r'$\sigma_{2}$ [MPa]')

            # axs[1][0].plot(dy_1_abaqus, label='ABAQUS', color='k',  marker='.',markersize=2.2, alpha=0.65)
            # axs[1][0].plot(ds_1_pred, label='ANN',color='r',  marker='.',markersize=2.2, alpha=0.5)
            # axs[1][0].set(xlabel=r'$t$', ylabel=r'$\dot{\sigma}_{1}$ [MPa/s]')

            # axs[1][1].plot(dy_2_abaqus, label='ABAQUS', color='k',  marker='.',markersize=2.2, alpha=0.65)
            # axs[1][1].plot(ds_2_pred, label='ANN',color='r',  marker='.',markersize=2.2, alpha=0.5)
            # axs[1][1].set(xlabel=r'$t$', ylabel=r'$\dot{\sigma}_{2}$ [MPa/s]')

            # axs[2][0].plot(dy_1_abaqus*de_1_abaqus, label='ABAQUS', color='k',  marker='.',markersize=2.2, alpha=0.65)
            # axs[2][0].plot(ds_1_pred*de_1_abaqus, label='ANN',color='r',  marker='.',markersize=2.2, alpha=0.5)
            # axs[2][0].set(xlabel=r'$t$', ylabel=r'$\dot{\sigma}_{1}\dot{\varepsilon}_{1}$')

            # axs[2][1].plot(dy_2_abaqus*de_2_abaqus, label='ABAQUS', color='k',  marker='.',markersize=2.2, alpha=0.65)
            # axs[2][1].plot(ds_2_pred*de_2_abaqus, label='ANN',color='r',  marker='.',markersize=2.2, alpha=0.5)
            # axs[2][1].set(xlabel=r'$t$', ylabel=r'$\dot{\sigma}_{2}\dot{\varepsilon}_{2}$')

            # handles, labels = axs[0][1].get_legend_handles_labels()
            # fig.legend(handles, labels, loc='lower center',ncol=2)

            # for ax in fig.axes:

            #     ax.tick_params(axis='x', labelcolor='black', length=6)
            #     ax.tick_params(axis='y', labelcolor='black',length=6)
            #     ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
            
            #     ax.xaxis.set_major_locator(MaxNLocator(6))
            #     ax.yaxis.set_major_locator(MaxNLocator(6)) 
            
            plt.savefig(os.path.join(DIR,k_test,'plots', f'{TRIAL}_el-{ELEM}_b.png'), format="png", dpi=600, bbox_inches='tight')
            plt.clf()
            plt.close(fig)

# for i, file in enumerate(file_list):

#     str = file.split('/')[-1][:-4]
#     if '_x_' in str:
#         expr = '_x_'
#     elif '__' in str:
#         expr = '__'
#     else:
#         expr = '_y_'
    
#     trial = str.split(expr)[0]
#     elem = str.split(expr)[-1]

#     ex_var_abaqus = df_list[i]['e_xx']
#     ey_var_abaqus = df_list[i]['e_yy']
#     exy_var_abaqus = df_list[i]['e_xy']
#     sx_var_abaqus = df_list[i]['s_xx']
#     sy_var_abaqus = df_list[i]['s_yy']
#     sxy_var_abaqus = df_list[i]['s_xy']

#     e_1_abaqus = df_list[i]['e_1']
#     e_2_abaqus = df_list[i]['e_2']

#     s_1_abaqus = df_list[i]['s_1']
#     s_2_abaqus = df_list[i]['s_2']

#     s_1_pred = df_list[i]['s_1_pred']
#     s_2_pred = df_list[i]['s_2_pred']

#     sx_pred_var = df_list[i]['s_xx_pred']
#     sy_pred_var = df_list[i]['s_yy_pred']
#     sxy_pred_var = df_list[i]['s_xy_pred']

#     de_1_abaqus = df_list[i]['de_1']
#     de_2_abaqus = df_list[i]['de_2']

#     dy_1_abaqus = df_list[i]['dy_1']
#     dy_2_abaqus = df_list[i]['dy_2']

#     ds_1_pred = df_list[i]['ds_1']
#     ds_2_pred = df_list[i]['ds_2']

#     fig , axs = plt.subplots(1,3)
#     #fig.set_size_inches(16, 9)
#     fig.set_size_inches(set_size('esaform',subplots=(1, 3)))
#     fig.subplots_adjust(bottom=0.28, wspace=0.35)
#     fig.suptitle(r'' + trial.replace('_','\_') + ': element \#' + elem, fontsize=9)

#     axs[0].plot(ex_var_abaqus, sx_var_abaqus, label='ABAQUS', color='k', marker='.', markersize=0.75)
#     axs[0].plot(ex_var_abaqus, sx_pred_var, label='ANN', color='r', marker='.', markersize=0.9, alpha=0.65)
#     #axs[0].plot(ex_var_abaqus, sx_pred_var_2, label='ANN(D)',color='b')
#     axs[0].set(xlabel=r'$\varepsilon_{xx}$', ylabel=r'$\sigma_{xx}$ [MPa]')
#     #axs[0].add_artist(set_anchored_text(mse_x,r2_x,mse_x_2,r2_x_2))
#     #axs[0].set_title(r'$\text{MSE}=%0.3f$' % (mse_x), fontsize=11)
#     axs[1].plot(ey_var_abaqus, sy_var_abaqus, label='ABAQUS', color='k', marker='.', markersize=0.75)
#     axs[1].plot(ey_var_abaqus, sy_pred_var, label='ANN', color='r', marker='.', markersize=0.9, alpha=0.65)
#     #axs[1].plot(ey_var_abaqus, sy_pred_var_2, label='ANN(D)',color='b')
#     axs[1].set(xlabel=r'$\varepsilon_{yy}$', ylabel=r'$\sigma_{yy}$ [MPa]')
#     #axs[1].add_artist(set_anchored_text(mse_y,r2_y,mse_y_2,r2_y_2))
#     #axs[1].set_title(r'$\text{MSE}=%0.3f$' % (mse_y), fontsize=11)
#     axs[2].plot(exy_var_abaqus, sxy_var_abaqus, label='ABAQUS', color='k', marker='.', markersize=0.75)
#     axs[2].plot(exy_var_abaqus, sxy_pred_var, label='ANN', color='r', marker='.', markersize=0.9, alpha=0.65)
#     #axs[2].plot(exy_var_abaqus, sxy_pred_var_2, label='ANN(D)',color='b')
#     axs[2].set(xlabel=r'$\varepsilon_{xy}$', ylabel=r'$\tau_{xy}$ [MPa]')
#     #axs[2].add_artist(set_anchored_text(mse_xy,r2_xy,mse_xy_2,r2_xy_2))
#     #axs[2].set_title(r'$\text{MSE}=%0.3f$' % (mse_xy), fontsize=11)
    
#     handles, labels = axs[0].get_legend_handles_labels()
#     fig.legend(handles, labels, loc='lower center',ncol=2)

#     for ax in fig.axes:

#         ax.tick_params(axis='x', labelcolor='black', length=6)
#         ax.tick_params(axis='y', labelcolor='black',length=6)
#         ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    
#         ax.xaxis.set_major_locator(MaxNLocator(6))
#         ax.yaxis.set_major_locator(MaxNLocator(6)) 

#     #plt.show()
#     plt.savefig(os.path.join(TEST_PATH,'data', '_1.png'), format="png", dpi=600, bbox_inches='tight')
#     plt.savefig(DIR + trial + expr + elem + "_1.png", format="png", dpi=600, bbox_inches='tight')
#     plt.close(fig)
#     plt.clf()

#     print('saving %i'% i)

#     fig , axs = plt.subplots(3,2)
#     #fig.set_size_inches(16, 9)
#     fig.set_size_inches(set_size('esaform',subplots=(2, 2)))
#     fig.subplots_adjust(bottom=0.1, wspace=0.35)
#     fig.suptitle(r'' + trial.replace('_','\_') + ': element \#' + elem, fontsize=9)

#     axs[0][0].plot(e_1_abaqus, s_1_abaqus, label='ABAQUS', color='k',  marker='.',markersize=2.2, alpha=0.65)
#     axs[0][0].plot(e_1_abaqus, s_1_pred, label='ANN',color='r',  marker='.',markersize=2.2, alpha=0.5)
#     axs[0][0].set(xlabel=r'$\varepsilon_{1}$', ylabel=r'$\sigma_{1}$ [MPa]')

#     axs[0][1].plot(e_2_abaqus, s_2_abaqus, label='ABAQUS', color='k',  marker='.',markersize=2.2, alpha=0.65)
#     axs[0][1].plot(e_2_abaqus, s_2_pred, label='ANN',color='r',marker='.',markersize=2.2, alpha=0.5)
#     axs[0][1].set(xlabel=r'$\varepsilon_{2}$', ylabel=r'$\sigma_{2}$ [MPa]')

#     axs[1][0].plot(dy_1_abaqus, label='ABAQUS', color='k',  marker='.',markersize=2.2, alpha=0.65)
#     axs[1][0].plot(ds_1_pred, label='ANN',color='r',  marker='.',markersize=2.2, alpha=0.5)
#     axs[1][0].set(xlabel=r'$t$', ylabel=r'$\dot{\sigma}_{1}$ [MPa/s]')

#     axs[1][1].plot(dy_2_abaqus, label='ABAQUS', color='k',  marker='.',markersize=2.2, alpha=0.65)
#     axs[1][1].plot(ds_2_pred, label='ANN',color='r',  marker='.',markersize=2.2, alpha=0.5)
#     axs[1][1].set(xlabel=r'$t$', ylabel=r'$\dot{\sigma}_{2}$ [MPa/s]')

#     axs[2][0].plot(dy_1_abaqus*de_1_abaqus, label='ABAQUS', color='k',  marker='.',markersize=2.2, alpha=0.65)
#     axs[2][0].plot(ds_1_pred*de_1_abaqus, label='ANN',color='r',  marker='.',markersize=2.2, alpha=0.5)
#     axs[2][0].set(xlabel=r'$t$', ylabel=r'$\dot{\sigma}_{1}\dot{\varepsilon}_{1}$')

#     axs[2][1].plot(dy_2_abaqus*de_2_abaqus, label='ABAQUS', color='k',  marker='.',markersize=2.2, alpha=0.65)
#     axs[2][1].plot(ds_2_pred*de_2_abaqus, label='ANN',color='r',  marker='.',markersize=2.2, alpha=0.5)
#     axs[2][1].set(xlabel=r'$t$', ylabel=r'$\dot{\sigma}_{2}\dot{\varepsilon}_{2}$')

#     handles, labels = axs[0][1].get_legend_handles_labels()
#     fig.legend(handles, labels, loc='lower center',ncol=2)

#     for ax in fig.axes:

#         ax.tick_params(axis='x', labelcolor='black', length=6)
#         ax.tick_params(axis='y', labelcolor='black',length=6)
#         ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    
#         ax.xaxis.set_major_locator(MaxNLocator(6))
#         ax.yaxis.set_major_locator(MaxNLocator(6)) 
    
#     #plt.show()
#     plt.savefig(DIR + trial + expr + elem + "_2.png", format="png", dpi=600, bbox_inches='tight')
#     plt.close(fig)
#     plt.clf()

    # print('\rProcessing image %i of %i' % (i+1,len(file_list)), end='')

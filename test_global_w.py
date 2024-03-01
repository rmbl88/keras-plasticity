import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import torchmetrics
from constants import *
import os
import pandas as pd
from cycler import cycler
from matplotlib.ticker import MaxNLocator
from matplotlib.offsetbox import AnchoredText
import glob
from tqdm import tqdm

def get_stats(var):
    return (np.mean(var),np.median(var),max(var),min(var))

def set_anchored_text(mean_e,median_e,max_e,min_e,frameon=True,loc='upper right'):
    at = AnchoredText(f'Mean: {np.round(mean_e,3)}\nMedian: {np.round(median_e,3)}\nMax.: {np.round(max_e,3)}\nMin.: {np.round(min_e,3)}', loc=loc, frameon=frameon,prop=dict(fontsize=6.5))
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
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1]) * 1.55

    return (fig_width_in, fig_height_in)


df_list = []

RUN = 'solar-planet-147'
#RUN = 'fine-rain-207'
DIR = f'outputs/crux-plastic_sbvf_abs_direct/val/{RUN}/'
#DIR = f'outputs/sbvfm_indirect_crux_gru/val/{RUN}/'

mech_tests = []
for r, d, f in os.walk(DIR):
    for folder in d:
        mech_tests.append(folder)

mech_tests = {k: None for k in mech_tests}

for k_test,v in mech_tests.items():
    
    TEST_PATH = os.path.join(DIR, k_test)

    file_list = glob.glob(os.path.join(TEST_PATH, 'data', f'*global_w.csv'))

    mech_tests[k_test] = {1:pd.read_csv(file) for file in file_list}

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
            NVFS = len([i for i in df.columns.to_list() if 'w_ext' in i])

            ux = TRIAL.split('_')[0][1:]
            uy = TRIAL.split('_')[1][1:]

            TITLE = f'$u_x={{{ux}}}$, $u_y={{{uy}}}$'
            
            # ex_var_abaqus = df['e_xx'].values
            # ey_var_abaqus = df['e_yy'].values
            # exy_var_abaqus = df['e_xy'].values
            # sx_var_abaqus = df['s_xx'].values
            # sy_var_abaqus = df['s_yy'].values
            # sxy_var_abaqus = df['s_xy'].values

            # sx_pred_var = df['s_xx_pred'].values
            # sy_pred_var = df['s_yy_pred'].values
            # sxy_pred_var = df['s_xy_pred'].values

            # err_x = np.abs(sx_var_abaqus - sx_pred_var)
            # err_y = np.abs(sy_var_abaqus - sy_pred_var)
            # err_xy = np.abs(sxy_var_abaqus - sxy_pred_var)
        
            # var_aba = [(ex_var_abaqus,sx_var_abaqus),(ey_var_abaqus,sy_var_abaqus),(exy_var_abaqus,sxy_var_abaqus)]
            # var_ann = [(ex_var_abaqus,sx_pred_var),(ey_var_abaqus,sy_pred_var),(exy_var_abaqus,sxy_pred_var)]
            # var_err = [err_x, err_y, err_xy]
            # var_lbl = [r'$\sigma_{xx}$', r'$\sigma_{yy}$', r'$\tau_{xy}$']


            fig , axs = plt.subplots(2,NVFS)
            fig.set_size_inches(set_size(468.0,subplots=(2, NVFS)))
            fig.subplots_adjust(bottom=0.15, wspace=0.38, hspace=0.38)
            
            line_labels = [r'$W_\mathrm{int}^*$ - ABAQUS',r'$W_\mathrm{int}^*$ - Dir-RNN',r'$W_\mathrm{ext}$']
            colors = ['k','royalblue','seagreen']
           
            errs=[]
            
            for i, ax in enumerate(fig.axes):
                if i<3:
                    vars = [var for var in df.columns.to_list()[1:] if f'{i}' in var]
                    data = [df[var].values for var in vars]
                    errs.append((np.abs(data[0]-data[-1]),np.abs(data[1]-data[-1])))
                
                    for k, label, color in zip(data,line_labels, colors):
                        ax.plot(k, label=label, color=color)
                    
                    ax.set_xlabel(r'Time stages')
                    ax.set_ylabel(r'VF-%i $W^*$ [J]'%(i+1))

                    # ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
            
                    ax.xaxis.set_major_locator(MaxNLocator(6))
                    ax.yaxis.set_major_locator(MaxNLocator(6)) 

                else:
                    
                    ax.plot(errs[i-3][0], color="k", marker='8', lw=0.65, markersize=2.25, markerfacecolor='white',markeredgewidth=0.35)
                    ax.plot(errs[i-3][1], color="royalblue", marker='s', lw=0.65, markersize=2.25, markerfacecolor='white',markeredgewidth=0.35)
                    
                    ax.set(xlabel=r'Time stages', ylabel=r'Abs. error [J]')
                    #ax.add_artist(set_anchored_text(*get_stats(var_err[i-3])))
                    # Choose the interval to space out xticks
                    _min = np.min(df.index.values)
                    diff = 24
                    # Decide positions of ticks
                    ticks = np.arange(_min, np.max(df.index.values), diff)
                    # Also, choose labels of the ticks
                    ax.set_xticks(ticks=ticks)
                    #ax.set_xlim(np.min(df.index.values), np.max(df.index.values))
           
            handles, labels = axs[0][0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower center',ncol=3)
        
            plt.savefig(os.path.join(DIR,k_test,'plots', f'{TRIAL}_global-W.png'), format="png", dpi=600, bbox_inches='tight')
            plt.clf()
            plt.close(fig)
        


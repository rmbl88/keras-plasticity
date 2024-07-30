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

DIR = 'one_elem'

ann_models = []
for r, d, f in os.walk(DIR):
    for folder in d:
        if 'Ind-RNN' not in folder:
            ann_models.append(folder)

# mech_tests = {k: None for k in mech_tests}

# for k_test,v in mech_tests.items():
    
#     file_list = glob.glob(os.path.join(TEST_PATH, 'data', f'*.csv'))

#     mech_tests[k_test] = {file.split('-')[-1][:-4]:pd.read_csv(file) for file in file_list}

matplotlib.use('Agg')
plt.rcParams.update(PARAMS)

default_cycler = (cycler(color=["#ef476f","#118ab2","#073b4c"]))

plt.rc('axes', prop_cycle=default_cycler)
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')

colors=["#DB222A","#3581B8","#62AB37",'#ffb703']
colors_umat=['#0C8346','#ff7d00','#8037AB','#034BFF']

color_dict = dict(zip(ann_models,colors))
colors_umat_dict = dict(zip(ann_models,colors_umat))

for model in ann_models:
    mech_tests = []
    for r, d, f in os.walk(os.path.join(DIR,model)):
        for file in f:
            if 'csv' in file:
                mech_tests.append(file)
    
    uniax_tests = {k.split('_')[0]: None for k in mech_tests}
    directions = ['xx','yy','xy']

    for i, (test, v) in enumerate(uniax_tests.items()):
        uniax_tests[test] = pd.read_csv(os.path.join(DIR,model,mech_tests[i]),sep=',',index_col=False)  
    
    # var_aba = [(ex_var_abaqus,sx_var_abaqus),(ey_var_abaqus,sy_var_abaqus),(exy_var_abaqus,sxy_var_abaqus)]
    # var_ann = [(ex_var_abaqus,sx_pred_var),(ey_var_abaqus,sy_pred_var),(exy_var_abaqus,sxy_pred_var)]
    # var_err = [err_x, err_y, err_xy]
    var_lbl = [r'$\sigma_{xx}$', r'$\sigma_{yy}$', r'$\tau_{xy}$']

    for j in range(2):

        fig , axs = plt.subplots(1,3)
        fig.set_size_inches(set_size(468.0,subplots=(1, 3)))
        fig.subplots_adjust(bottom=0.28, wspace=0.38, hspace=0.38, top=0.88)
        
        axes_labels = [(r'$\varepsilon_{xx}$',r'$\sigma_{xx}$ [MPa]'),(r'$\varepsilon_{yy}$',r'$\sigma_{yy}$ [MPa]'),(r'$\varepsilon_{xy}$',r'$\tau_{xy}$ [MPa]')]
        
        for i, ax in enumerate(fig.axes):
            
            if i<3:
                k = i
            else:
                k = i-3

            strain = [col for col in uniax_tests[directions[k]].columns if 'e_'+directions[k] in col]
            stress = [col for col in uniax_tests[directions[k]].columns if 's_'+directions[k] in col and len(col.split('_'))<=2]
            stress_pred = [col for col in uniax_tests[directions[k]].columns if directions[k]+'_pred' in col]
            stress_umat = [col for col in uniax_tests[directions[k]].columns if directions[k]+'_umat' in col and 'err' not in col]

            #if i<3:
            if j == 0:

                ax.plot(uniax_tests[directions[k]][strain], uniax_tests[directions[k]][stress], label=r'\textbf{ABAQUS}', color='k')
                ax.plot(uniax_tests[directions[k]][strain], uniax_tests[directions[k]][stress_pred], label=r'\textbf{%s}' % model, color=color_dict[model])
                ax.plot(uniax_tests[directions[k]][strain], uniax_tests[directions[k]][stress_umat], label=r'\textbf{%s-UMAT}' % model, color=colors_umat_dict[model])
                ax.set_xlabel(axes_labels[i][0])
                ax.set_ylabel(axes_labels[i][1])

                ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        
                ax.xaxis.set_major_locator(MaxNLocator(6))
                ax.yaxis.set_major_locator(MaxNLocator(6)) 

            else:
                err_pred = np.abs(uniax_tests[directions[k]][stress].values-uniax_tests[directions[k]][stress_pred].values)
                err_umat = np.abs(uniax_tests[directions[k]][stress].values-uniax_tests[directions[k]][stress_umat].values)
                
                ax.plot(err_pred, color=color_dict[model], marker='8', lw=0.65, markersize=2.25, markerfacecolor='white',markeredgewidth=0.35)
                ax.plot(err_umat, color=colors_umat_dict[model], marker='s', lw=0.65, markersize=2.25, markerfacecolor='white',markeredgewidth=0.35)
                #axs[i].plot(df.index.values, vars[i], color="black", lw=0.1)
                #ax.set(xlabel=r'Time stages', ylabel=r'%s - \textbf{Abs. error} [MPa]' % (var_lbl[i-3]))
                ax.set(xlabel=r'Time stages', ylabel=f'{var_lbl[i]} - Abs. error [MPa]')
                #ax.add_artist(set_anchored_text(*get_stats(var_err[i-3])))
                # Choose the interval to space out xticks
                #_min = np.min(df.index.values)
                #diff = 24
                # Decide positions of ticks
                #ticks = np.arange(_min, np.max(df.index.values), diff)
                # Also, choose labels of the ticks
                #ax.set_xticks(ticks=ticks)
                #ax.set_xlim(np.min(df.index.values), np.max(df.index.values))
        
        if j == 0:
            handles, labels = axs[0].get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
        else:
            by_label = dict(zip(labels[1:], handles[1:]))
        fig.legend(by_label.values(), by_label.keys(), loc='lower center',ncol=len(by_label.keys()))

        plt.savefig(os.path.join(DIR, model, f'{DIR}_{model}_{j}.pdf'), format="pdf", dpi=600, bbox_inches='tight')
        plt.clf()
        plt.close(fig)
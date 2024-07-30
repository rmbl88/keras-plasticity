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

from plot_utils import set_size

def get_stats(var):
    return (np.mean(var),np.median(var),max(var),min(var))

def set_anchored_text(mean_e,median_e,max_e,min_e,frameon=True,loc='upper right',title=None, fontsize=6.5):
    if title == None:
        t_str = ''
    else:
        t_str = title + '\n'
    at = AnchoredText(f'{t_str}Mean: {np.round(mean_e,3)}\nMedian: {np.round(median_e,3)}\nMax.: {np.round(max_e,3)}\nMin.: {np.round(min_e,3)}', loc=loc, frameon=frameon,prop=dict(fontsize=fontsize))
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    at.patch.set_linewidth(0.55)
    return at

# def set_size(width, fraction=1, subplots=(1, 1)):
#     """Set figure dimensions to avoid scaling in LaTeX.

#     Parameters
#     ----------
#     width: float or string
#             Document width in points, or string of predined document type
#     fraction: float, optional
#             Fraction of the width which you wish the figure to occupy
#     subplots: array-like, optional
#             The number of rows and columns of subplots.
#     Returns
#     -------
#     fig_dim: tuple
#             Dimensions of figure in inches
#     """
#     if width == 'thesis':
#         width_pt = 426.79135
#     elif width == 'beamer':
#         width_pt = 307.28987
#     elif width == 'esaform':
#         width_pt = 535.896
#     else:
#         width_pt = width

#     # Width of figure (in pts)
#     fig_width_pt = width_pt * fraction
#     # Convert from pt to inches
#     inches_per_pt = 1 / 72.27

#     # Golden ratio to set aesthetic figure height
#     # https://disq.us/p/2940ij3
#     golden_ratio = (5**.5 - 1) / 2

#     # Figure width in inches
#     fig_width_in = fig_width_pt * inches_per_pt
#     # Figure height in inches
#     fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1]) * 1.55

#     return (fig_width_in, fig_height_in)


df_list = []

#RUN = 'solar-planet-147'
#RUN = 'fine-rain-207'
RUN = '20240724-233559'
#DIR = f'outputs/indirect_crux_gru/val/{RUN}/'
#DIR = f'outputs/crux-plastic-jm_sbvf_abs_direct/val/{RUN}/'
DIR = f'outputs/vfm_training_gru/val/{RUN}/'
#DIR = f'outputs/sbvfm_indirect_crux_gru/val/{RUN}/'

mech_tests = []
for r, d, f in os.walk(DIR):
    for folder in d:
        mech_tests.append(folder)

mech_tests = {k: None for k in mech_tests}

if 'one_elem_001' in mech_tests.keys():
    mech_tests.update({'one_elem_001': {'1_elem_xx': None, '1_elem_yy': None, '1_elem_xy': None}})

for k_test,v in mech_tests.items():
    
    TEST_PATH = os.path.join(DIR, k_test)

    file_list = glob.glob(os.path.join(TEST_PATH, 'data', f'*.csv'))

    if k_test == 'one_elem_001':
        files = {file.split('\\')[-1].split('-')[0]:pd.read_csv(file) for file in file_list}
        
        mech_tests.update({'one_elem_001': files})

    else:

        mech_tests[k_test] = {file.split('-')[-1][:-4]:pd.read_csv(file) for file in file_list}

#matplotlib.use('Agg')
plt.rcParams.update(PARAMS)

default_cycler = (cycler(color=["#ef476f","#118ab2","#073b4c"]))

plt.rc('axes', prop_cycle=default_cycler)
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')

err = torchmetrics.MeanAbsoluteError()
r2 = torchmetrics.R2Score()

TAGS = ['x15_y15_', 'plate_hole', 's_shaped', 'sigma_shaped']
#TAGS = ['plate_hole']

for k_test, elems in (pbar_1 := tqdm(mech_tests.items(), bar_format=FORMAT_PBAR, leave=False)):
    
    pbar_1.set_description(f'Processing trial - {k_test}')

    if k_test in TAGS or k_test == 'one_elem_001':

        if k_test == 'plate_hole':

            plate_sol = pd.read_csv(os.path.join(DIR, k_test, 'plate_solution.csv'))

            cols = plate_sol.columns.to_list()

            lbls = [r'\textbf{Dir-RNN}', r'\textbf{ABAQUS}', r'\textbf{Reference}']
            colors = ['royalblue', 'k', 'k']
            markers = ['.','o','-']

            fig , ax = plt.subplots(1,1)
            fig.set_size_inches(set_size(234,subplots=(1, 1)))
            fig.subplots_adjust(bottom=0.2)

            for i, col in enumerate(reversed(cols[1:])):
                ax.plot(plate_sol[cols[0]], plate_sol[col], markers[2-i], 
                        label=lbls[2-i],
                        color=colors[2-i], 
                        ms=3, 
                        markerfacecolor="None" if 'pred' not in col else colors[2-i], 
                        markeredgewidth=0.5 if 'pred' not in col else 0.35)
            
            ax.set_ylabel(r'$\sigma_{xx~(x=0)}$ [MPa]')
            ax.set_xlabel(r'$\mathrm{Distance}~y$ [mm]')
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower center', ncol=len(lbls))
            
            plt.savefig(os.path.join(DIR, k_test, 'plate_solution.pdf'), format="pdf", dpi=600, bbox_inches='tight')
            plt.clf()
            plt.close(fig)
    
        for elem, df in (pbar_2 := tqdm(elems.items(), bar_format=FORMAT_PBAR, leave=False)):
            
            pbar_2.set_description(f'Plotting elem-{elem}')

            TRIAL = k_test
            ELEM = elem

            if k_test == 'one_elem_001':
                
                u = ELEM.split('_')[2]
                dir = ELEM.split('_')[3]
                TITLE = f'$u_{{{dir}}}={{{u}}}$'

                ex_var_abaqus = df['e_xx'].values
                ey_var_abaqus = df['e_yy'].values
                exy_var_abaqus = df['e_xy'].values
                sx_var_abaqus = df['s_xx'].values
                sy_var_abaqus = df['s_yy'].values
                sxy_var_abaqus = df['s_xy'].values

                sx_pred_var = df['s_xx_pred'].values
                sy_pred_var = df['s_yy_pred'].values
                sxy_pred_var = df['s_xy_pred'].values

                err_x = df['abs_err_sx'].values
                err_y = df['abs_err_sy'].values
                err_xy = df['abs_err_sxy'].values

                if 's_xx_umat' in df.columns:
                    sx_umat = df['s_xx_umat'].values
                    sy_umat = df['s_yy_umat'].values
                    sxy_umat = df['s_xy_umat'].values

                    err_x_umat = df['abs_err_sxx_umat'].values
                    err_y_umat = df['abs_err_syy_umat'].values
                    err_xy_umat = df['abs_err_sxy_umat'].values

                    var_umat = [(ex_var_abaqus,sx_umat),(ey_var_abaqus,sy_umat),(exy_var_abaqus,sxy_umat)]
                    var_err_umat = [err_x_umat, err_y_umat, err_xy_umat]

                var_aba = [(ex_var_abaqus,sx_var_abaqus),(ey_var_abaqus,sy_var_abaqus),(exy_var_abaqus,sxy_var_abaqus)]
                var_ann = [(ex_var_abaqus,sx_pred_var),(ey_var_abaqus,sy_pred_var),(exy_var_abaqus,sxy_pred_var)]
                var_err = [err_x, err_y, err_xy]
                

            else:
                ux = TRIAL.split('_')[0][1:]
                uy = TRIAL.split('_')[1][1:]

                TITLE = f'$u_x={{{ux}}}$, $u_y={{{uy}}}$ - element {ELEM}'
                
                ex_var_abaqus = df['e_xx'].values
                ey_var_abaqus = df['e_yy'].values
                exy_var_abaqus = df['e_xy'].values
                sx_var_abaqus = df['s_xx'].values
                sy_var_abaqus = df['s_yy'].values
                sxy_var_abaqus = df['s_xy'].values

                sx_pred_var = df['s_xx_pred'].values
                sy_pred_var = df['s_yy_pred'].values
                sxy_pred_var = df['s_xy_pred'].values

                err_x = df['abs_err_sx'].values
                err_y = df['abs_err_sy'].values
                err_xy = df['abs_err_sxy'].values

                var_err = [err_x, err_y, err_xy]
        
            var_aba = [(ex_var_abaqus,sx_var_abaqus),(ey_var_abaqus,sy_var_abaqus),(exy_var_abaqus,sxy_var_abaqus)]
            var_ann = [(ex_var_abaqus,sx_pred_var),(ey_var_abaqus,sy_pred_var),(exy_var_abaqus,sxy_pred_var)]
            
            var_lbl = [r'$\sigma_{xx}$', r'$\sigma_{yy}$', r'$\tau_{xy}$']

            fig , axs = plt.subplots(2,3)
            fig.set_size_inches(set_size(468.0,subplots=(2, 3)))
            fig.subplots_adjust(bottom=0.15, wspace=0.38, hspace=0.38)
            
            axes_labels = [(r'$\varepsilon_{xx}$',r'$\sigma_{xx}$ [MPa]'),(r'$\varepsilon_{yy}$',r'$\sigma_{yy}$ [MPa]'),(r'$\varepsilon_{xy}$',r'$\tau_{xy}$ [MPa]')]
           
            for i, ax in enumerate(fig.axes):
                if i<3:
                    ax.plot(var_aba[i][0], var_aba[i][1], label=r'\textbf{ABAQUS}', color='k')
                    ax.plot(var_ann[i][0], var_ann[i][1], label=r'\textbf{Dir-RNN}', color='royalblue')
                    if (k_test == 'one_elem_001') and ('s_xx_umat' in df.columns):
                        ax.plot(var_umat[i][0], var_umat[i][1], label=r'\textbf{UMAT-RNN}', color='green')
                    ax.set_xlabel(axes_labels[i][0])
                    ax.set_ylabel(axes_labels[i][1])

                    ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
            
                    ax.xaxis.set_major_locator(MaxNLocator(6))
                    ax.yaxis.set_major_locator(MaxNLocator(6)) 

                else:
                    
                    ax.plot(df.index.values, var_err[i-3], color="royalblue", marker='8', lw=0.65, markersize=2.25, markerfacecolor='white',markeredgewidth=0.35)
                    if (k_test == 'one_elem_001') and ('s_xx_umat' in df.columns):
                        ax.plot(df.index.values, var_err_umat[i-3], color="green", marker='8', lw=0.65, markersize=2.25, markerfacecolor='white',markeredgewidth=0.35)                        

                    ax.set(xlabel=r'Time stages', ylabel=r'Abs. error [MPa]')
                    if k_test == 'one_elem_001':
                        ax.add_artist(set_anchored_text(*get_stats(var_err[i-3]), title='Dir-RNN', fontsize=5.75))
                        if 's_xx_umat' in df.columns:
                            ax.add_artist(set_anchored_text(*get_stats(var_err_umat[i-3]), loc='upper left', title='UMAT-RNN', fontsize=5.75))
                    else:
                        ax.add_artist(set_anchored_text(*get_stats(var_err[i-3])))

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
        
            if k_test == 'one_elem_001':
                plt.savefig(os.path.join(DIR,k_test,'plots', f'{ELEM}.pdf'), format="pdf", dpi=600, bbox_inches='tight')
            else:
                plt.savefig(os.path.join(DIR,k_test,'plots', f'{TRIAL}_el-{ELEM}_a.pdf'), format="pdf", dpi=600, bbox_inches='tight')
            plt.clf()
            plt.close(fig)

        

            


        


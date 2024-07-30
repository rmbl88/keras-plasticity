import copy
import matplotlib
from matplotlib import gridspec
from matplotlib import ticker
import matplotlib.tri as tri
import matplotlib.pyplot as plt
from cycler import cycler
from constants import *
import os
import torch
import numpy as np
import mpl_toolkits.axisartist as axisartist
import matplotlib.cm as mcm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from functions import get_principal, rotate_tensor

def get_tri_mesh(nodes: np.array, connectivity: np.array):

    def quads_to_tris(quads):
        tris = [[None for j in range(3)] for i in range(2*len(quads))]
        for i in range(len(quads)):
            j = 2*i
            n0 = quads[i][0]
            n1 = quads[i][1]
            n2 = quads[i][2]
            n3 = quads[i][3]
            tris[j][0] = n0
            tris[j][1] = n1
            tris[j][2] = n2
            tris[j + 1][0] = n2
            tris[j + 1][1] = n3
            tris[j + 1][2] = n0
        return tris

    tris = quads_to_tris(connectivity)
    triangulation = tri.Triangulation(nodes[:,0], nodes[:,1], tris)

    return triangulation

def contour_plot(triangulation, field, ax=None, **kwargs):

        if not ax: ax = plt.gca()
        pc = ax.tricontourf(triangulation, field.flatten(), **kwargs)

        # This is the fix for the white lines between contour levels
        for c in pc.collections:
            #c.set_edgecolor("face")
            c.set_rasterized(True)

        #ax.autoscale()

        return pc

def set_size(width, fraction=1, subplots=(1, 1), multiplier=1.55):
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
    elif width == 'elsevier':
        width_pt = 468.0
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
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1]) * multiplier

    return (fig_width_in, fig_height_in)

# def plot_rafr(rafr: dict, dir: str):

#     plt.rcParams.update(PARAMS)

#     default_cycler = (cycler(color=["#ef476f","#118ab2","#073b4c"]))

#     plt.rc('axes', prop_cycle=default_cycler)
#     plt.rc('xtick', labelsize='x-small')
#     plt.rc('ytick', labelsize='x-small')

#     fig , axs = plt.subplots(2,2, sharey='row')
#     fig.set_size_inches(set_size('elsevier',subplots=(2, 3), multiplier=1))
#     fig.subplots_adjust(bottom=0.16, wspace=0.15, hspace=0.5, left=0.2, right=0.8)

#     t_stages = list(rafr.keys())

#     for i, ax in enumerate(axs.flatten()):
#         ax.plot(rafr[t_stages[i]]['coord'], rafr[t_stages[i]]['rafr'], label='RAFR - DIR-RNN', color='royalblue', zorder=4)
#         ax.set_title(r'$F_a=%.3f$ kN' % (rafr[t_stages[i]]['axial_f'].item()/1000), fontsize=7.5, pad=3.75)
#         ax.axhline(1.0, linestyle='--', color='k', lw=0.35, zorder=3)
        
#         if 's_shaped' in dir:
#             ax.axvline(-8, linestyle='--', color='k', lw=0.3, zorder=2)
#             ax.axvline(8, linestyle='--', color='k', lw=0.3, zorder=2)
#             ax.set_xticks([np.min(rafr[t_stages[i]]['coord']).item(),-25,-8,0,8,25,np.max(rafr[t_stages[i]]['coord']).item()])
#             ax.set_xticklabels([int(np.min(rafr[t_stages[i]]['coord']).item()),-25,-8,0,8,25,int(np.max(rafr[t_stages[i]]['coord']).item())], ha='center')
#             ax.set_xlim([np.min(rafr[t_stages[i]]['coord'])-5,np.max(rafr[t_stages[i]]['coord'])+5])

#         elif 'sigma_shaped' in dir:
#             ax.set_xticks([np.min(rafr[t_stages[i]]['coord']).item(),-30,-15,0,15,30,np.max(rafr[t_stages[i]]['coord']).item()])
#             ax.set_xticklabels([int(np.min(rafr[t_stages[i]]['coord']).item()),-30,-15,0,15,30,int(np.max(rafr[t_stages[i]]['coord']).item())], ha='center')
#             ax.set_xlim([np.min(rafr[t_stages[i]]['coord'])-5,np.max(rafr[t_stages[i]]['coord'])+5])

#         elif 'plate-hole':
#             ax.set_xlim([np.min(rafr[t_stages[i]]['coord'])-0.5,np.max(rafr[t_stages[i]]['coord'])+0.5])

#         ax.fill_between(x=np.linspace(np.min(rafr[t_stages[i]]['coord'])-10, np.max(rafr[t_stages[i]]['coord'])+10,5), y1=1.1, y2=0.9, color='lightgray',  interpolate=True, alpha=.55, zorder=1)

#         ax.set_ylim([0.5, 1.5])
#         ax.set_yticks([0.5,1.0,1.5], labels=['0.5','1.0','1.5'])

#     handles, labels = axs[0][0].get_legend_handles_labels()
#     fig.legend(handles, labels, loc='lower center')

#     plt.savefig(os.path.join(dir, 'rafr.pdf'), format="pdf", dpi=600, bbox_inches='tight')
#     plt.clf()
#     plt.close(fig)

def plot_rafr(rafr_objs: dict, specimen: str, dir: str):

    plt.rcParams.update(PARAMS)

    default_cycler = (cycler(color=["#DB222A","#3581B8","#62AB37","#0E131F"]))

    plt.rc('axes', prop_cycle=default_cycler)
    plt.rc('xtick', labelsize='x-small')
    plt.rc('ytick', labelsize='x-small')

    fig , axs = plt.subplots(3,2, sharey='row')
    fig.set_size_inches(set_size('elsevier', subplots=(3, 2), multiplier=0.625))
    fig.subplots_adjust(bottom=0.115, wspace=0.15, hspace=0.675, left=0.2, right=0.8)

    t_stages = list(rafr_objs[list(rafr_objs.keys())[0]].keys())

    for i, ax in enumerate(axs.flatten()):
     
        for j, (model, rafr) in enumerate(rafr_objs.items()):
            ax.plot(rafr[t_stages[i]]['coord'], rafr[t_stages[i]]['rafr'], label=f'{model}', zorder=4)
            ax.set_title(r'$F_a=%.3f$ kN $\left(t_n=%i\right)$' % (rafr[t_stages[i]]['axial_f'].item()/1000, t_stages[i]), fontsize=7.5, pad=3.75)
        ax.axhline(1.0, linestyle='--', color='k', lw=0.35, zorder=3)
        
        if specimen == 's_shaped':
            ax.axvline(-8, linestyle='--', color='k', lw=0.3, zorder=2)
            ax.axvline(8, linestyle='--', color='k', lw=0.3, zorder=2)
            ax.set_xticks([np.min(rafr[t_stages[i]]['coord']).item(),-25,-8,0,8,25,np.max(rafr[t_stages[i]]['coord']).item()])
            ax.set_xticklabels([int(np.min(rafr[t_stages[i]]['coord']).item()),-25,-8,0,8,25,int(np.max(rafr[t_stages[i]]['coord']).item())], ha='center')
            ax.set_xlim([np.min(rafr[t_stages[i]]['coord'])-5,np.max(rafr[t_stages[i]]['coord'])+5])
        
        elif specimen == 'd_shaped':
            ax.set_xticks([np.min(rafr[t_stages[i]]['coord']).item(),-24,-12,0,12,24,np.max(rafr[t_stages[i]]['coord']).item()])
            ax.set_xticklabels([int(np.min(rafr[t_stages[i]]['coord']).item()),-24,-12,0,12,24,int(np.max(rafr[t_stages[i]]['coord']).item())], ha='center')
            ax.set_xlim([np.min(rafr[t_stages[i]]['coord'])-3.5,np.max(rafr[t_stages[i]]['coord'])+3.5])

        elif specimen == 'sigma_shaped':
            ax.set_xticks([np.min(rafr[t_stages[i]]['coord']).item(),-30,-15,0,15,30,np.max(rafr[t_stages[i]]['coord']).item()])
            ax.set_xticklabels([int(np.min(rafr[t_stages[i]]['coord']).item()),-30,-15,0,15,30,int(np.max(rafr[t_stages[i]]['coord']).item())], ha='center')
            ax.set_xlim([np.min(rafr[t_stages[i]]['coord'])-5,np.max(rafr[t_stages[i]]['coord'])+5])

        elif specimen == 'plate-hole':
            ax.set_xlim([np.min(rafr[t_stages[i]]['coord'])-0.5,np.max(rafr[t_stages[i]]['coord'])+0.5])

        ax.fill_between(x=np.linspace(np.min(rafr[t_stages[i]]['coord'])-10, np.max(rafr[t_stages[i]]['coord'])+10,5), y1=1.1, y2=0.9, color='lightgray',  interpolate=True, alpha=.55, zorder=1)

        ax.set_ylim([0.35, 1.65])
        ax.set_yticks([0.5,1.0,1.5], labels=['0.5','1.0','1.5'])

    handles, labels = axs[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(rafr_objs.keys()))

    plt.savefig(os.path.join(dir, f'rafr-{specimen}.pdf'), format="pdf", dpi=600, bbox_inches='tight')
    plt.clf()
    plt.close(fig)

def plot_lode_triax(stress_data: torch.Tensor, dir: str):
    
    plt.rcParams.update(PARAMS)

    default_cycler = (cycler(color=["#ef476f","#118ab2","#073b4c"]))

    plt.rc('axes', prop_cycle=default_cycler)
    plt.rc('xtick', labelsize='x-small')
    plt.rc('ytick', labelsize='x-small')

    fig , ax = plt.subplots(1,1)
    fig.set_size_inches(set_size(234,subplots=(1, 1)))
    fig.subplots_adjust(bottom=0.2)

    s_h = ((stress_data[:,:,0] + stress_data[:,:,1]) / 3).unsqueeze(-1)
    s_vm = (stress_data[:,:,0].square() - stress_data[:,:,0] * stress_data[:,:,1] + stress_data[:,:,1].square() + 3 * stress_data[:,:,2].square()).pow(0.5)
        
    triax = s_h[:,:,0] / s_vm

    lode = 1-(2/torch.pi)*torch.acos(-(27/2)*triax*(triax.square()-1/3))

    t = torch.linspace(-1,1,100)
    l = 1-(2/torch.pi)*torch.acos(-(27/2)*t*(t.square()-1/3))
    ax.plot(l,t, linestyle='--', color='k', lw=0.35, zorder=3)
    ax.axhline(0.0, linestyle=':', color='lightgray', lw=0.35, zorder=1)
    ax.axvline(0.0, linestyle=':', color='lightgray', lw=0.35, zorder=2)

    colors = ['k', 'royalblue']
    markers = ['o', '^']
    marker_sizes = [6,3]
    labels = ['Abaqus', 'Dir-RNN']
    for i in range(stress_data.shape[0]):
        ax.scatter(lode[i], triax[i], s=marker_sizes[i], marker=markers[i], facecolors='none', edgecolors=colors[i], linewidths=0.45 if i==0 else 0.35, zorder=i+4, label=labels[i])

    ax.set_ylim([-1,1])
    ax.set_yticks([-1,-2/3,-1/3,0,1/3,2/3,1],[r'$-1$',r'$\large{\sfrac{-2}{3}}$',r'$\large{\sfrac{-1}{3}}$',r'$0$',r'$\large{\sfrac{1}{3}}$',r'$\large{\sfrac{2}{3}}$',r'$1$'])
    ax.set_ylabel(r'Triaxiality $(\mathcal{\eta})$')
    ax.set_xlabel(r'Lode angle parameter $\bar{(\theta)}$')
    lgnd = plt.legend(loc='best', handletextpad=0.1)

    lgnd.legendHandles[1]._sizes = [6]

    plt.savefig(os.path.join(dir, 'lode_triax.pdf'), format="pdf", dpi=600, bbox_inches='tight')
    plt.clf()
    plt.close(fig)

def plot_scatter_yield(stress: np.array, strain: np.array, peeq: np.array, dir: str, test: str, nodes: np.array, connectivity: np.array, t_domain: bool = False):

    plt.rcParams.update(PARAMS)

    #plt.rc('axes', prop_cycle=default_cycler)
    plt.rc('xtick', labelsize='x-small')
    plt.rc('ytick', labelsize='x-small')

    #default_cycler = (cycler(color=["#ef476f","#118ab2","#073b4c"]))

    # Calulacting yield stress
    e_0 = (160/565)**(1/0.26)
    yield_stress = 565*(e_0+peeq)**(0.26)

    # Normalizing stress by the yield stress
    s_normal = stress/yield_stress

    # Calculating principal strains and stresses
    e = strain
    e[:,-1] *= 0.5
    eps_princ, _ = get_principal(e.reshape(-1,e.shape[-1]))
    eps_princ = eps_princ.reshape(*strain.shape[:2],2)
    eps_princ = np.sort(eps_princ, axis=2)

   
    # Setting plotting figure
    fig = plt.figure()
    fig.set_size_inches(6.03,2.04)
    # Creating a 1-row 2-column left and a 1-row 1-column right
    gs_left = gridspec.GridSpec(1, 2)
    gs_right = gridspec.GridSpec(1, 1)

    # Adjusting container positions
    gs_left.update(right=0.65, top=0.95)
    gs_right.update(left=0.68, top=0.95)
    gs_left.update(wspace=0.384)

    # Populating axes objects    
    #axs = [fig.add_subplot(gs_left[0,0]), fig.add_subplot(gs_left[0,1]), fig.add_subplot(gs_right[0,0])]
    axs = [fig.add_subplot(gs_left[0,0]), fig.add_subplot(gs_left[0,1])]
    #axs[-1].axis('off')

    # Setting axes aspect ratio
    axs[1].set_aspect('equal')
    axs[-1].set_aspect('equal')   

    # Setting colorbar limits and labels
    cbar_min = np.min(peeq)
    cbar_max= np.max(peeq)
    cbarlabels = np.linspace(cbar_min, cbar_max, num=8+1)
    
    # Setting up the colormap and color levels
    cmap = copy.copy(mcm.jet)
    levels  = np.linspace(cbar_min, cbar_max,256)
    norm = matplotlib.colors.BoundaryNorm(levels, 256)

    # Setting up mesh
    triangulation = get_tri_mesh(nodes,connectivity)

    if t_domain:
        # Extrapolating from centroids to nodes
        var_nodal = np.ones((4,1)) @ np.zeros([connectivity.shape[0],1]).reshape(-1,1,1)
        # Average values at nodes
        var_avg = np.array([np.mean(var_nodal[np.isin(connectivity,j)],0) for j in range(len(nodes))])
        #axs[-1].patch.set_color('.25')
        #pc = contour_plot(triangulation, var_avg, ax=axs[-1], colors='darkgray')
    else:
        # Extrapolating from centroids to nodes
        var_nodal = np.ones((4,1)) @ peeq[-1].reshape(-1,1,1)

        # Average values at nodes
        var_avg = np.array([np.mean(var_nodal[np.isin(connectivity,j)],0) for j in range(len(nodes))])

        # Plotting peeq field
        #pc = contour_plot(triangulation, var_avg, ax=axs[-1], cmap=cmap, levels=levels, norm=norm, vmin=cbar_min, vmax=cbar_max)

    # Masking elastic points
    mask = peeq == 0.0

    # Getting max data limit
    if test == 'plate_hole':
        data_max = round(5*round(np.max(eps_princ) / 5, 3), 3)
    else:
        data_max = np.ceil(np.max(np.abs(eps_princ))*10)/10

    #x_lims = axs[-1].get_xlim()
    #y_lims = axs[-1].get_ylim()

    #max_x = np.max(x_lims)
    #max_y = np.max(y_lims)
    
    # if test == 's_shaped' or test == 'sigma_shaped':

    #     if np.max([max_x,max_y]) == max_x:
    #         axs[-1].set_ylim([-max_x,max_x])
    #     else:
    #         axs[-1].set_xlim([-max_y,max_y])

    # elif test == 'd_shaped':

    #     extent_x = x_lims[1]-x_lims[0]
    #     extent_y = y_lims[1]-y_lims[0]

    #     delta_extent = abs(extent_x - extent_y) / 2

    #     if extent_x > extent_y:
    #         axs[-1].set_ylim([y_lims[0]-delta_extent,y_lims[1]+delta_extent])
    #     else:
    #         axs[-1].set_xlim([x_lims[0]-delta_extent,x_lims[1]+delta_extent])

    axs[0].scatter(eps_princ[:,:,0], eps_princ[:,:,1], s=0.75, c=peeq.flatten(), zorder=2, alpha=1, vmin=cbar_min, vmax=cbar_max, cmap=cmap, marker='o', rasterized=True, edgecolors='none')

    if test == 'plate_hole' or test == 'sigma_shaped':
        axs[0].set_xlim([-data_max*1.2, data_max*1.2])
        axs[0].set_ylim([0, data_max*1.2])
    else:
        axs[0].set_xlim([-data_max, data_max])
        axs[0].set_ylim([0, data_max])

    axs[0].set_aspect((data_max - (-data_max)) / data_max)
    
    axs[0].axhline(0.0, color='black', lw=0.2, zorder=1)
    axs[0].axvline(0.0, color='black', lw=0.2, zorder=1)

    axs[0].plot([0.5, 1], [0, 1], transform=axs[0].transAxes, color='black', lw=0.2, zorder=1)
    axs[0].plot([0.5, 0], [0, 1], transform=axs[0].transAxes, color='black', lw=0.2, zorder=1)
    axs[0].plot([0.5, 0], [0, 0.5], transform=axs[0].transAxes, color='black', lw=0.2, zorder=1)
    axs[0].plot([0.5, 0.25], [0, 1], transform=axs[0].transAxes, color='black', lw=0.2, zorder=1)

    if test == 'plate_hole':
        axs[0].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        axs[0].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    kwargs = {'transform':axs[0].transAxes, 
              'fontsize':4.5, 
              'horizontalalignment': 'center', 
              'bbox':dict(facecolor='white', boxstyle='square,pad=0.1', ec='none')
    }

    axs[0].axes.text(0.15,0.05, r'plane strain''\n'r'compression', **kwargs)
    axs[0].axes.text(0.15,0.325, r'uniaxial''\n'r'compression', **kwargs)
    axs[0].axes.text(0.275,0.875, r'uniaxial''\n'r'tension', **kwargs)
    axs[0].axes.text(0.85,0.8, r'equibiaxial''\n'r'tension', **kwargs)
    axs[0].axes.text(0.12,0.8, r'shear', **kwargs)
    axs[0].axes.text(0.5,0.8,  r'plane strain''\n'r'tension', **kwargs)

    axs[0].set_xlabel(r'$\large{\bm{\varepsilon}_{2}}$')
    axs[0].set_ylabel(r'$\large{\bm{\varepsilon}_{1}}$')
    
    axs[1].scatter(s_normal[:,:,0][mask[:,:,0]].flatten(), s_normal[:,:,1][mask[:,:,0]].flatten(), s=0.75, c='gray', zorder=2, alpha=1, marker='o', rasterized=True, edgecolors='none')
    s=axs[1].scatter(s_normal[:,:,0][~mask[:,:,0]].flatten(), s_normal[:,:,1][~mask[:,:,0]].flatten(), s=0.75, c=peeq[~mask].flatten(), zorder=2, alpha=1, vmin=cbar_min, vmax=cbar_max, cmap=cmap, marker='o', rasterized=True, edgecolors='none')

    axs[1].axhline(0.0, color='black', lw=0.2, zorder=1)
    axs[1].axvline(0.0, color='black', lw=0.2, zorder=1)

    data_max = np.ceil(np.max(np.abs(s_normal[:,:,:2])))

    x = np.linspace(-data_max,data_max,100)
    y = np.linspace(-data_max,data_max,100)
    z = [0.0, 0.2, 0.4, 0.6, 0.8]

    X,Y = np.meshgrid(x, y, indexing="ij")

    for zi in z:
        vm = np.sqrt(np.square(X) - X * Y + np.square(Y) + 3*np.square(zi))
        axs[1].contour(X,Y,vm, zorder=3, colors='black', linewidths=0.3, levels=[1.0])

    axs[1].set_xlim([-data_max, data_max])
    axs[1].set_ylim([-data_max, data_max])
    axs[1].set_xlabel(r'$\large{\sfrac{\bm{\sigma}_{xx}}{\sigma_{Y}}}$')
    axs[1].set_ylabel(r'$\large{\sfrac{\bm{\sigma}_{yy}}{\sigma_{Y}}}$', labelpad=0.35)
    
    inset_axes = (1.05, 0.025, 0.02, 0.8)
    if test == 'plate_hole':
        # fmt = ticker.ScalarFormatter(useMathText=True)
        # fmt.set_powerlimits((0,0))
        fmt = '%.1e'
    else:
        fmt = '%.3f'

    #cbar=fig.colorbar(s, ticks=cbarlabels, cax=axs[-1].inset_axes(inset_axes), format=fmt)
    cbar=fig.colorbar(s, ticks=cbarlabels, cax=axs[1].inset_axes(inset_axes), format=fmt)
    
    cbar.ax.yaxis.set_offset_position('right')
    cbar.ax.set_title(r'$\bm{\varepsilon}_{\mathrm{eq}}^{\mathrm{p}}$', pad=10, horizontalalignment='left', fontsize=PARAMS['axes.labelsize'])
    cbar.outline.set_linewidth(0.45)
    cbar.minorticks_off()
    cbar.ax.yaxis.set_tick_params(pad=5, colors='black', width=0.45, labelsize=7)
    cbar.ax.yaxis.set_ticks_position('right')
    cbar.ax.yaxis.offsetText.set_fontsize(7)    

    plt.savefig(os.path.join(dir,'yield.pdf'), format="pdf", dpi=600, bbox_inches='tight')
    plt.clf()
    plt.close(fig)
  
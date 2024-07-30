import matplotlib
import numpy as np
from functions import *
from constants import *
import matplotlib.cm as mcm
import matplotlib.collections as mc
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox, AnchoredText, TextArea
import matplotlib.tri as tri
from tqdm import tqdm
import gc
import matplotlib.gridspec as gridspec

def create_dir(dir: str, root_dir: str):

    ROOT_DIR = root_dir
    DIR = os.path.join(ROOT_DIR, dir)

    try:    
        os.makedirs(DIR)        
    except FileExistsError:
        pass

    return DIR

def plot_fields(nodes, connectivity, fields, out_dir, tag, vf=None, ann_run=None):
    
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

    def quatplot(x, y, elements, field, ax=None, **kwargs):

        if not ax: ax=plt.gca()
        xy = np.c_[x,y]
        verts= xy[elements]
        pc = mc.PolyCollection(verts, **kwargs)
        pc.set_array(field)
        ax.add_collection(pc)
        ax.autoscale()

        return pc
    
    def set_anchored_text(mean_e,median_e,max_e,min_e,vf=None,frameon=True,loc='upper right', bbox_to_anchor=None, transform=None):
        if vf == None:
            at = AnchoredText(f'Mean: {np.round(mean_e,3)}\nMedian: {np.round(median_e,3)}\nMax.: {np.round(max_e,3)}\nMin.: {np.round(min_e,3)}', prop=dict(fontsize=PARAMS_CONTOUR['legend.fontsize']), loc=loc, frameon=frameon)
        else:
            if bbox_to_anchor != None:
                at = AnchoredText(f'VF - {vf}\nMean: {np.round(mean_e,3)}\nMedian: {np.round(median_e,3)}\nMax.: {np.round(max_e,3)}\nMin.: {np.round(min_e,3)}', frameon=frameon,prop=dict(fontsize=PARAMS_CONTOUR['legend.fontsize']), bbox_to_anchor=bbox_to_anchor, bbox_transform=transform)
            else:
                at = AnchoredText(f'VF - {vf}\nMean: {np.round(mean_e,3)}\nMedian: {np.round(median_e,3)}\nMax.: {np.round(max_e,3)}\nMin.: {np.round(min_e,3)}', loc=loc, frameon=frameon,prop=dict(fontsize=PARAMS_CONTOUR['legend.fontsize']))
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        at.patch.set_linewidth(0.55)
        return at
    
    #matplotlib.use('TkAgg')
    plt.ioff()
    plt.rcParams.update(PARAMS_CONTOUR)  

    triangulation = get_tri_mesh(nodes,connectivity)
    
    for t, vars in fields.items():
        
        for var, fields_ in (pbar := tqdm(vars.items(), bar_format=FORMAT_PBAR, leave=False)):
            pbar.set_description(f'Saving countour plot -> {tag}t{t}_{var}')
            
            fig = plt.figure()
            fig.set_size_inches(19.2,10.8)

            if tag == 's_shaped' or tag =='sigma_shaped':
                # create a 1-row 3-column container as the left container
                gs_left = gridspec.GridSpec(1, 2)
                # create a 1-row 1-column grid as the right container
                gs_right = gridspec.GridSpec(1, 2)

                axs = [fig.add_subplot(gs_left[0,0]), fig.add_subplot(gs_left[0,1]), fig.add_subplot(gs_right[0,0]), fig.add_subplot(gs_right[0,1])]
            else:

                # create a 1-row 3-column container as the left container
                gs_left = gridspec.GridSpec(1, 2)
                # create a 1-row 1-column grid as the right container
                gs_right = gridspec.GridSpec(1, 1)

                if tag == 'd_shaped':
                    gs_err = gridspec.GridSpec(1,1)
                    axs = [fig.add_subplot(gs_left[0,0]), fig.add_subplot(gs_left[0,1]), fig.add_subplot(gs_right[0,0]), fig.add_subplot(gs_err[0,0])]
                else:
                    axs = [fig.add_subplot(gs_left[0,0]), fig.add_subplot(gs_left[0,1]), fig.add_subplot(gs_right[0,0])]

            # gs_left.update(right=0.61)
            # gs_right.update(left=0.672)

            if tag=='s_shaped' or tag =='sigma_shaped':
                gs_left.update(right=0.45, top=0.62, bottom=0.25)
                gs_right.update(left=0.45, top=0.62, bottom=0.25, right=0.9)
                gs_left.update(wspace=0.2)
                gs_right.update(wspace=0)

            elif tag =='d_shaped':
                gs_left.update(right=0.57, top=0.62, bottom=0.25)
                gs_right.update(left=0.525, top=0.62, bottom=0.25)
                gs_err.update(left=0.765, top=0.62, bottom=0.25, right=0.85)
                gs_left.update(wspace=0.2)

            else:
                gs_left.update(right=0.61)
                gs_right.update(left=0.695)
                gs_left.update(wspace=0.35)

            #gs_right.update(wspace=0.025)

            #fig.subplots_adjust(wspace=0.15)
            
            if var == 'sxx_t':
                cb_str = r'$\boldsymbol{\sigma}_{xx}~[\mathrm{MPa}]$'
            elif var == 'syy_t':
                cb_str = r'$\boldsymbol{\sigma}_{yy}~[\mathrm{MPa}]$'
            elif var == 'sxy_t':
                cb_str = r'$\boldsymbol{\tau}_{xy}~[\mathrm{MPa}]$'
            elif var == 's1':
                cb_str = r'$\boldsymbol{\sigma}_{11}~[\mathrm{MPa}]$'
            elif var == 's2':
                cb_str = r'$\boldsymbol{\sigma}_{22}~[\mathrm{MPa}]$'
            elif var == 'mises':
                cb_str = r'$\boldsymbol{\sigma}_{vM}~[\mathrm{MPa}]$'
            elif var == 'ivw_xx':
                cb_str = r'$\boldsymbol{W}^{int}_{xx}~[\mathrm{J}]$'
            elif var == 'ivw_yy':
                cb_str = r'$\boldsymbol{W}^{int}_{yy}~[\mathrm{J}]$'
            elif var == 'ivw_xy':
                cb_str = r'$\boldsymbol{W}^{int}_{xy}~[\mathrm{J}]$'
            elif var == 'ivw':
                cb_str = r'$\boldsymbol{W}^{int}~[\mathrm{J}]$'
            else:
                cb_str=''

            for i, (k,v) in enumerate(fields_.items()):

                if v is not None:
                    # Extrapolating from centroids to nodes
                    var_nodal = np.ones((4,1)) @ v.reshape(-1,1,1)

                    # Average values at nodes
                    var_avg = np.array([np.mean(var_nodal[np.isin(connectivity,j)],0) for j in range(len(nodes))])

                    # if i == 0 and k == 'abaqus':
                    #     # Defining contour levels
                    #     cbar_min = np.min(var_avg)
                    #     cbar_max = np.max(var_avg)
                    # elif i > 1 and k!= 'ann':
                    #     cbar_min = np.min(var_avg)
                    #     cbar_max = np.max(var_avg)

                    if i != 1 and k != 'ann':
                        # Defining contour levels
                        cbar_min = np.min(var_avg)
                        cbar_max = np.max(var_avg)
                   
                        
                    cmap = copy.copy(mcm.jet)

                    #if k == 'ann':
                    if k=='abaqus' or k == 'ann':
                        cmap.set_under('white')
                        cmap.set_over('black')
                        CB_EXTEND = 'both'
                    else:
                        CB_EXTEND = 'neither'

                    levels  = np.linspace(cbar_min, cbar_max,256)
                    norm = matplotlib.colors.BoundaryNorm(levels, 256)

                    if not np.all(var_avg==0):
                        pc = contour_plot(triangulation, var_avg, ax=axs[i], cmap=cmap, levels=levels, norm=norm, vmin=cbar_min, vmax=cbar_max, extend=CB_EXTEND)
                    else:
                        pc = 0
                        cb = 0
                        continue

                    # pc = quatplot(nodes[:,0], nodes[:,1], connectivity, v.reshape(-1), ax=axs[i], edgecolor="face", linewidths=0.1, cmap=cmap,snap=True, norm=norm)
            
                    axs[i].axis('off')
                    axs[i].set_aspect('equal')
                    
                    cbarlabels = np.linspace(cbar_min, cbar_max, num=10, endpoint=True)

                    #fmt ='%.3e'
                    fmt = '%.3f'
                    if k == 'abaqus':
                        cb_str_ = r'\textbf{Abaqus}' + '\n' + cb_str
                    elif k == 'ann':
                        cb_str_ = rf'\textbf{{{ann_run}}}' + '\n' + cb_str
                        #axs[i].text(-0.045, 0.91, cb_str_, horizontalalignment='right', verticalalignment='center', transform=axs[i].transAxes,fontsize=PARAMS_CONTOUR['legend.fontsize'])
                    
                    elif k == 'err':
                        f_var = cb_str.split('~')[0][1:]
                        cb_str_ = r'\textbf{Abs. error}' + '\n' + r'$\boldsymbol{\delta}{%s}~[\mathrm{MPa}]$' % (f_var)
                        fmt = '%.3f'

                        v_mean = np.mean(var_avg)
                        v_median = np.median(var_avg)
                        v_max = np.max(var_avg)
                        v_min = np.min(var_avg)

                        if tag == 's_shaped' or tag =='sigma_shaped':
                            axs[-1].axis('off')
                            # Shrink current axis by 20%
                            box = axs[-1].get_position()
                            axs[-1].set_position([box.x0 - 0.035, box.y0, box.width * 0.35, box.height])
                            arts = set_anchored_text(v_mean,v_median,v_max,v_min,vf, loc='upper left')
                            axs[-1].add_artist(arts)
                        elif tag == 'd_shaped':
                            axs[-1].axis('off')
                            arts = set_anchored_text(v_mean,v_median,v_max,v_min,vf, loc='center left', bbox_to_anchor=(0.5,0.5), transform=axs[i].transAxes)
                            axs[-1].add_artist(arts)
                            box = axs[-1].get_position()
                            axs[-1].set_position([box.x0 - 0.035, box.y0, box.width * 0.35, box.height])
                        else:
                            arts = set_anchored_text(v_mean,v_median,v_max,v_min,vf)
                            axs[i].add_artist(arts)

                    #if i!=1:
                    if tag == 's_shaped' or tag =='sigma_shaped':
                        inset_axes = (-0.085, 0.025, 0.03, 0.8)
                    else:
                        inset_axes = (-0.05, 0.025, 0.02, 0.8)
                    cb = fig.colorbar(pc, cax=axs[i].inset_axes(inset_axes),ticks=cbarlabels,format=fmt)
                    cb.ax.set_title(cb_str_, pad=15, horizontalalignment='right')
                    cb.outline.set_linewidth(1)
                    cb.minorticks_off()
                    cb.ax.yaxis.set_tick_params(pad=7.5, colors='black', width=1,labelsize=PARAMS_CONTOUR['axes.labelsize'])
                    cb.ax.yaxis.set_ticks_position('left')
                    
            # fig.tight_layout()
            # plt.show()
            #fig.savefig(os.path.join(out_dir,f'{tag}_t{t}_{var}_cont.png'), format="png", dpi=600, bbox_inches='tight')
            if vf == None:
                fig.savefig(os.path.join(out_dir,f'{tag}_t{t}_{var}_cont.pdf'), format="pdf", dpi=600, bbox_inches='tight')
                if var == 'mises':
                    arts.remove()
                    for n, ax in enumerate(axs):
                        if n < 3:
                            extent = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
                            # if tag == 'd_shaped':
                            #     extent_err = axs[-1].get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
                            #     extent.x1 = extent_err.x1
                            out_ = create_dir('single', out_dir)
                            fig.savefig(os.path.join(out_, f'{tag}_t{t}_{var}_ax_{n}.pdf'), bbox_inches=extent.expanded(1.01,1.01), format="pdf", dpi=600)
                            ax.remove()

            else:
                fig.savefig(os.path.join(out_dir,f'{tag}_t{t}_{var}_vf-{vf}_cont.pdf'), format="pdf", dpi=600, bbox_inches='tight')
                
            plt.clf()
            # plt.close(fig)
            del pc, cb, axs, fig, var_nodal, var_avg
            gc.collect()
            

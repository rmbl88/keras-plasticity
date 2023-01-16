import matplotlib
import numpy as np
from functions import *
from constants import *
import matplotlib.cm as mcm
import matplotlib.collections as mc
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import ticker
import matplotlib.tri as tri

# # plots a finite element mesh
# def plot_fem_mesh(nodes_x, nodes_y, elements,ax=None):
#     for element in elements:
#         x = [nodes_x[element[i]] for i in range(len(element))]
#         y = [nodes_y[element[i]] for i in range(len(element))]
#         ax.fill(x, y, edgecolor='black', fill=False)

def plot_fields(nodes, connectivity, fields):
    
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
        ax.autoscale()

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

    plt.rcParams.update(PARAMS_CONTOUR)  
    cmap = mcm.jet

    triangulation = get_tri_mesh(nodes,connectivity)
    
    for t, vars in fields.items():
        
        for var, fields_ in vars.items():
        
            n_subplots = len(vars[var].keys())
            fig, axs = plt.subplots(1,n_subplots)
            fig.set_size_inches(12,6.75)
            
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

            for i, (k,v) in enumerate(fields_.items()):

                if v is not None:
                    # Extrapolating from centroids to nodes
                    var_nodal = np.ones((4,1)) @ v.reshape(-1,1,1)
                    # Average values at nodes
                    var_avg = np.array([np.mean(var_nodal[np.isin(connectivity,j)],0) for j in range(len(nodes))])

                    # Defining contour levels
                    cbar_min = np.min(var_avg)
                    cbar_max = np.max(var_avg)
                        
                    levels  = np.linspace(cbar_min, cbar_max,256)
                    norm = matplotlib.colors.BoundaryNorm(levels, 256)

                
                    pc = contour_plot(triangulation, var_avg, ax=axs[i], cmap=cmap, levels=levels, norm=norm, vmin=cbar_min, vmax=cbar_max)
                   
                    # pc = quatplot(nodes[:,0], nodes[:,1], connectivity, v.reshape(-1), ax=axs[i], edgecolor="face", linewidths=0.1, cmap=cmap,snap=True, norm=norm)
            
                    axs[i].axis('off')
                    axs[i].set_aspect('equal')
                    
                    cbarlabels = np.linspace(cbar_min, cbar_max, num=13, endpoint=True)

                    fmt ='%.3e'
                    if k == 'abaqus':
                        cb_str_ = r'\textbf{Abaqus}' + '\n' + cb_str
                    elif k == 'ann':
                        cb_str_ = r'\textbf{ANN}' + '\n' + cb_str
                    elif k == 'err':
                        f_var = cb_str.split('~')[0][1:]
                        cb_str_ = r'\textbf{Abs. error}' + '\n' + r'$\boldsymbol{\delta}_{%s}~[\mathrm{MPa}]$' % (f_var)
                        fmt = '%.3f'

                    cb = fig.colorbar(pc, cax=axs[i].inset_axes((-0.05, 0.12, 0.02, 0.65)),ticks=cbarlabels,format=fmt)
                    cb.ax.set_title(cb_str_, pad = 15, horizontalalignment='right')
                    cb.ax.yaxis.set_tick_params(pad=7.5)
                    cb.ax.yaxis.set_ticks_position('left')

            fig.tight_layout()
            plt.show()

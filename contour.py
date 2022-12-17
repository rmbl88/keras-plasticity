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

# plots a finite element mesh
def plot_fem_mesh(nodes_x, nodes_y, elements,ax=None):
    for element in elements:
        x = [nodes_x[element[i]] for i in range(len(element))]
        y = [nodes_y[element[i]] for i in range(len(element))]
        ax.fill(x, y, edgecolor='black', fill=False)

def plot_fields(nodes, connectivity, fields, triangulation):

    def contour_plot(triangulation, field, ax=None, **kwargs):

        if not ax: ax = plt.gca()
        pc = ax.tricontourf(triangulation, field.flatten(), **kwargs)
        ax.autoscale()

        return pc

    n_subplots = len(fields.keys())

    fig, axs = plt.subplots(1,n_subplots)
    
    fig.set_size_inches(12,6.75)
        
    cmap = mcm.jet

    for i, (k, v) in enumerate(fields.items()):

        # Extrapolating from centroids to nodes
        var_nodal = np.ones((4,1)) @ v.reshape(-1,1,1)
        # Average values at nodes
        var_avg = np.array([np.mean(var_nodal[np.isin(connectivity,j)],0) for j in range(len(nodes))])

        # Defining contour levels
        cbar_min = np.min(var_avg)
        cbar_max = np.max(var_avg)
        levels  = np.linspace(np.floor(cbar_min), np.ceil(cbar_max),256)
        norm = matplotlib.colors.BoundaryNorm(levels, 256)

        pc = contour_plot(triangulation, var_avg, ax=axs[i], cmap=cmap, levels=levels, norm=norm, vmin=cbar_min, vmax=cbar_max)
    
        axs[i].axis('off')
        axs[i].set_aspect('equal')
        cbarlabels = np.linspace(np.floor(cbar_min), np.ceil(cbar_max), num=13, endpoint=True)
        cb = fig.colorbar(pc, cax=axs[i].inset_axes((-0.05, 0.12, 0.02, 0.65)),ticks=cbarlabels,format='%.3e')

        if k == 'sxx_t':
            cb_str = r'$\boldsymbol{\sigma}_{xx}~[\mathrm{MPa}]$'
        elif k == 'syy_t':
            cb_str = r'$\boldsymbol{\sigma}_{yy}~[\mathrm{MPa}]$'
        elif k == 'sxy_t':
            cb_str = r'$\boldsymbol{\tau}_{xy}~[\mathrm{MPa}]$'

        cb.ax.set_title(cb_str, pad = 15, horizontalalignment='right')
        cb.ax.yaxis.set_tick_params(pad=7.5)
        cb.ax.yaxis.set_ticks_position('left')


    fig.tight_layout()
    plt.show()

plt.rcParams.update(PARAMS_CONTOUR)

# Loading data
data=pq.ParquetDataset(os.path.join(TRAIN_MULTI_DIR,'processed','x05_y05_.parquet')).read_pandas().to_pandas()

# Importing mesh
mesh, connectivity, dof = read_mesh(TRAIN_MULTI_DIR)

# Extracting nodal coordinates and connectivity
nodes = mesh[:,1:]
connectivity = connectivity[:,1:] - 1

# Converting from quad mesh to triangular mesh
tris = quads_to_tris(connectivity)
triangulation = tri.Triangulation(nodes[:,0], nodes[:,1], tris)

# Field variables to plot
vars = ['sxx_t','syy_t','sxy_t']
field_dict = dict.fromkeys(vars)

for k, _ in field_dict.items():
    field_dict[k] =  data[data['inc']==119][k].values

plot_fields(nodes, connectivity, field_dict, triangulation)

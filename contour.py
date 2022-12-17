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
def plot_fem_mesh(nodes_x, nodes_y, elements):
    for element in elements:
        x = [nodes_x[element[i]] for i in range(len(element))]
        y = [nodes_y[element[i]] for i in range(len(element))]
        plt.fill(x, y, edgecolor='black', fill=False)

def showMeshPlot(nodes, connectivity, fields):

    n_subplots = len(fields.keys())

    x = nodes[:,0]
    y = nodes[:,1]

    def quatplot(x, y, elements, field, ax=None, **kwargs):

        if not ax: ax=plt.gca()
        xy = np.c_[x,y]
        verts= xy[elements]
        pc = mc.PolyCollection(verts, **kwargs)
        pc.set_array(field)
        ax.add_collection(pc)
        ax.autoscale()
        
        return pc


    fig, axs = plt.subplots(1,n_subplots)
    
    fig.set_size_inches(12,6.75)
    
    
    cmap = mcm.jet

    for i, (k, v) in enumerate(fields.items()):

        axs[i].set_aspect('equal')
        bounds  = np.linspace(min(v),max(v),13)
    
        pc = quatplot(x, y, np.asarray(connectivity), v, ax=axs[i], edgecolor="face", linewidths=0.25, cmap=cmap,snap=True, norm=None)
    
        axs[i].plot(x,y, ls="")
        axs[i].axis('off')
    
        cb = fig.colorbar(pc, cax=axs[i].inset_axes((-0.05, 0.12, 0.02, 0.65)), boundaries=bounds, ticks=bounds,format='%.3e')
        #cb.formatter.set_useMathText(True)
        if k == 'sxx_t':
            cb_str = r'$\boldsymbol{\sigma}_{xx}~[\mathrm{MPa}]$'
        elif k == 'syy_t':
            cb_str = r'$\boldsymbol{\sigma}_{yy}~[\mathrm{MPa}]$'
        elif k == 'sxy_t':
            cb_str = r'$\boldsymbol{\tau}_{xy}~[\mathrm{MPa}]$'

        cb.ax.set_title(cb_str, pad = 15, horizontalalignment='right')
        cb.ax.yaxis.set_tick_params(pad=7.5)
        
        #cb.ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        cb.ax.yaxis.set_ticks_position('left')


    fig.tight_layout()
    plt.show()

plt.rcParams.update(PARAMS_CONTOUR)
# plt.rc('text', usetex=True)
# plt.rc('text.latex', preamble=r"\usepackage{amsmath,newpxtext,newpxmath}")

data=pq.ParquetDataset(os.path.join(TRAIN_MULTI_DIR,'processed','x05_y05_.parquet')).read_pandas().to_pandas()

mesh, connectivity, dof = read_mesh(TRAIN_MULTI_DIR)

nodes = mesh[:,1:]
connectivity = connectivity[:,1:] - 1

vars = ['sxx_t','syy_t','sxy_t']
field_dict = dict.fromkeys(vars)

for k, _ in field_dict.items():
    field_dict[k] =  data[data['inc']==119][k].values

a=np.array([[0.25],[0.25],[0.25],[0.25]])@field_dict['sxy_t'].reshape(-1,1,1)
meanStress = np.zeros((nodes.shape[0],1))
for i in range(len(nodes)):
    mask = np.isin(connectivity,i)
    meanStress[i,:] = np.mean(a[mask],0) 

elements_all_tris = quads_to_tris(connectivity)
triangulation = tri.Triangulation(nodes[:,0], nodes[:,1], elements_all_tris)
#plot_fem_mesh(nodes[:,0], nodes[:,1], connectivity)
fig,ax = plt.subplots()
ax.tricontourf(triangulation, meanStress.flatten(),levels=256,cmap=mcm.jet,vmin=min(meanStress),vmax=max(meanStress),aspect='equal')
ax.set_aspect('equal')
plt.show()
#showMeshPlot(nodes, connectivity, field_dict)

print('hey')

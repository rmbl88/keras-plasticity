
from mesh_utils import (
    get_geom_limits, 
    read_mesh
)

import pandas as pd
import os
import numpy as np
import joblib

from vfm import (
    get_ud_vfs
)

import matplotlib
matplotlib.use('TKAgg')
import numpy as np
from functions import *
from constants import *
import matplotlib.cm as mcm
import matplotlib.collections as mc
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.offsetbox import AnchoredText
from matplotlib import ticker
import matplotlib.tri as tri
from tqdm import tqdm
import gc

def plot_fields(nodes, connectivity, fields, out_dir, var):
    
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
        pc = ax.tricontourf(triangulation, field.flatten(),**kwargs)

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
    
    def set_anchored_text(mean_e,median_e,max_e,min_e,frameon=True,loc='upper right'):
        at = AnchoredText(f'Mean: {np.round(mean_e,3)}\nMedian: {np.round(median_e,3)}\nMax.: {np.round(max_e,3)}\nMin.: {np.round(min_e,3)}', loc=loc, frameon=frameon,prop=dict(fontsize=PARAMS_CONTOUR['font.size']))
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        at.patch.set_linewidth(0.55)
        return at
    
  
    plt.ioff()
    plt.rcParams.update(PARAMS_CONTOUR)  

    triangulation = get_tri_mesh(nodes,connectivity)

    dirs = ['xx','yy','xy']
    
    for i, field in (pbar := tqdm(enumerate(fields[:]), bar_format=FORMAT_PBAR, leave=False)):
        
        if VFM_TYPE == 'sb':
            if var == 'disp':
                v_x = field[-1][::2]
                v_y = field[-1][1::2]
                field = torch.stack([v_x,v_y],1).squeeze(-1).numpy()
            if var == 'strain':
                field = field[:,-1].numpy()

        pbar.set_description(f'Saving countour plot -> v_{var}_{i}')

        n_subplots = field.shape[-1]
        fig, axs = plt.subplots(1,n_subplots)
        fig.set_size_inches(19.2 if n_subplots==3 else 4,2)
        fig.subplots_adjust(wspace=0.15)    

        for j in range(field.shape[-1]):    
            
            v = field[:,j]
            
            if var == 'disp':
                cb_str = r'$\boldsymbol{{u}}_{{{0}}}^{{*({1})}}$'.format(dirs[j],i+1)
            elif var == 'strain':
                cb_str = r'$\boldsymbol{{\varepsilon}}_{{{0}}}^{{*({1})}}$'.format(dirs[j],i+1)

            if v is not None:
                
                if var == 'strain':
                    # Extrapolating from centroids to nodes
                    var_nodal = np.ones((4,1)) @ v.reshape(-1,1,1)

                    # Average values at nodes
                    var_avg = np.array([np.mean(var_nodal[np.isin(connectivity,j)],0) for j in range(len(nodes))])

                    # Defining contour levels
                    cbar_min = np.min(field)
                    cbar_max = np.max(field)*1.05

                    if cbar_max == cbar_min:
                        cbar_max = 1.

                else:

                    var_avg = v.reshape(-1,1,1)

                     # Defining contour levels
                    cbar_min = np.min(field)
                    cbar_max = np.max(field)

                cmap = copy.copy(mcm.jet)

                levels  = np.linspace(cbar_min, cbar_max,256)
                norm = matplotlib.colors.BoundaryNorm(levels, 256)

                pc = contour_plot(triangulation, var_avg, ax=axs[j], cmap=cmap, levels=levels, norm=norm, vmin=cbar_min, vmax=cbar_max)
        
                axs[j].axis('off')
                axs[j].set_aspect('equal')
                
                if(j==0):
                    cbarlabels = np.linspace(cbar_min, cbar_max, num=6, endpoint=True)
                    fmt = '%.3f'
                    cb = fig.colorbar(pc, cax=axs[j].inset_axes((-0.075, 0.025, 0.03, 0.8)),ticks=cbarlabels,format=fmt)
                    cb.ax.set_title(cb_str, pad=11, horizontalalignment='right',size=10)
                    cb.outline.set_linewidth(0.45)
                    cb.minorticks_off()
                    cb.ax.yaxis.set_tick_params(pad=2.5, colors='black', width=0.45, labelsize=10, length=1.75)
                    cb.ax.yaxis.set_ticks_position('left')
                else:
                    axs[j].text(-0.13, 0.97, cb_str, horizontalalignment='center', verticalalignment='center', transform=axs[j].transAxes,fontsize=10)

        # fig.tight_layout()
        # plt.show()
        #fig.savefig(os.path.join(out_dir,f'v_{var}_{i}.png'), format="png", dpi=200, bbox_inches='tight')
        fig.savefig(os.path.join(out_dir,f'v_{var}_{i}.pdf'), format="pdf", dpi=600, bbox_inches='tight')
        plt.clf()
        # plt.close(fig)
        del pc, cb, axs, fig, var_avg
        gc.collect()



TRAIN_DIR = 'data/training_multi/crux-plastic/'
VFM_TYPE = 'ud'

# Reading mesh file
MESH, CONNECTIVITY, DOF = read_mesh(TRAIN_DIR)

# Defining geometry limits
X_MIN, X_MAX, Y_MIN, Y_MAX = get_geom_limits(MESH)

# Element centroids
CENTROIDS = pd.read_csv(os.path.join(TRAIN_DIR,'centroids.csv'), usecols=['cent_x','cent_y']).values

if VFM_TYPE == 'ud':  # User-defined VFM

    # Maximum dimensions of specimen
    WIDTH = X_MAX - X_MIN
    HEIGHT = Y_MAX - Y_MIN

    # Computing virtual fields
    TOTAL_VFS, V_DISP, V_STRAIN = get_ud_vfs(CENTROIDS, MESH[:,1:], WIDTH, HEIGHT)

elif VFM_TYPE == 'sb':
    V_DISP, V_STRAIN = joblib.load('test_vfs.pkl')

vf_dict = {
    'disp': V_DISP,
    'strain': V_STRAIN
}

for var, fields in vf_dict.items():
    plot_fields(MESH[:,1:], CONNECTIVITY[:,1:]-1, fields, 'vfs', var)



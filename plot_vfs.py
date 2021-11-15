import numpy as np
from numpy.core.numeric import empty_like, zeros_like
from constants import *
import matplotlib.pyplot as plt
import math
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm, ListedColormap

import sys

plt.rcParams.update(PARAMS)

rc = {"axes.spines.left" : False,
      "axes.spines.right" : False,
      "axes.spines.bottom" : False,
      "axes.spines.top" : False,
      "xtick.bottom" : False,
      "xtick.labelbottom" : False,
      "ytick.labelleft" : False,
      "ytick.left" : False}

plt.rcParams.update(rc)

 # Defining nodal coordinates
coords = np.linspace(0, int(LENGTH), 20)

x, y = np.meshgrid(coords, coords)

#nodes =  np.vstack(map(np.ravel, np.meshgrid(coords, coords))).T

#n_nodes = nodes.shape[0]
        
#x = nodes[:,0]        
#y = nodes[:,1]

zeros_ = np.zeros_like(x)

virtual_disp = {
            1: [x/LENGTH, zeros_],
            2: [zeros_, y/LENGTH],
            3: [zeros_, y*(np.square(x)-x*LENGTH)/LENGTH**3],
            4: [zeros_, np.sin(x*math.pi/LENGTH)*np.sin(y*math.pi/LENGTH)],
            5: [np.sin(y*math.pi/LENGTH) * np.sin(x*math.pi/LENGTH), zeros_],
            6: [x*y*(x-LENGTH)/LENGTH**3,zeros_],
            7: [np.square(x)*(LENGTH-x)*np.sin(math.pi*y/LENGTH)/LENGTH**3,zeros_],
            8: [zeros_, (LENGTH**3-x**3)*np.sin(math.pi*y/LENGTH)/LENGTH**3],
            9: [(x*LENGTH**2-x**3)*np.sin(math.pi*y/LENGTH)/LENGTH**3,zeros_],
            10: [(x*y*(x-LENGTH)/LENGTH**2)*np.sin(y*math.pi/LENGTH), zeros_],
            11: [zeros_, (x*y*(y-LENGTH)/LENGTH**2)*np.sin(x*math.pi/LENGTH)],
            12: [zeros_,x*y*(y-LENGTH)/LENGTH**3],
            13: [y**2*np.sin(x*math.pi/LENGTH)/LENGTH**2, zeros_],
            14: [zeros_, x**2*np.sin(y*math.pi/LENGTH)/LENGTH**2],
            15: [(x*y*(x-LENGTH)/LENGTH**2)*np.sin(x**2*y**2/LENGTH**4), zeros_],
            16: [np.sin(x*math.pi/LENGTH)/LENGTH, zeros_],
            17: [zeros_, np.sin(y*math.pi/LENGTH)/LENGTH],
            18: [np.sin(x**3*math.pi/LENGTH**3)/LENGTH**3, zeros_],
            19: [zeros_, np.sin(y**3*math.pi/LENGTH**3)/LENGTH**3],
                   
        }

formulas = {
            1: r"$u^{*~(%i)}=\begin{Bmatrix}\cfrac{x}{L} & 0\end{Bmatrix}$",
            2: r"$u^{*~(%i)}=\begin{Bmatrix}0 & \cfrac{y}{L}\end{Bmatrix}$",
            3: r"$u^{*~(%i)}=\begin{Bmatrix}0 & y\cfrac{x^2-xL}{L^3}\end{Bmatrix}$",
            4: r"$u^{*~(%i)}=\begin{Bmatrix}0 & \sin\left(\cfrac{x\pi}{L}\right)\sin\left(\cfrac{y\pi}{L}\right)\end{Bmatrix}$",
            5: r"$u^{*~(%i)}=\begin{Bmatrix}\sin\left(\cfrac{x\pi}{L}\right)\sin\left(\cfrac{y\pi}{L}\right) & 0\end{Bmatrix}$",
            6: r"$u^{*~(%i)}=\begin{Bmatrix}\cfrac{xy(x-L)}{L^3} & 0\end{Bmatrix}$",
            7: r"$u^{*~(%i)}=\begin{Bmatrix}\cfrac{x^2(L-x)}{L^3}\sin\left(\cfrac{y\pi}{L}\right) & 0\end{Bmatrix}$",
            8: r"$u^{*~(%i)}=\begin{Bmatrix} 0 & \cfrac{L^3-x^3}{L^3}\sin\left(\cfrac{y\pi}{L}\right)\end{Bmatrix}$",
            9: r"$u^{*~(%i)}=\begin{Bmatrix}\cfrac{xL^2-x^3}{L^3}\sin\left(\cfrac{y\pi}{L}\right) & 0\end{Bmatrix}$",
            10: r"$u^{*~(%i)}=\begin{Bmatrix}\cfrac{xy(x-L)}{L^2}\sin\left(\cfrac{y\pi}{L}\right) & 0\end{Bmatrix}$",
            11: r"$u^{*~(%i)}=\begin{Bmatrix}0 & \cfrac{xy(y-L)}{L^2}\sin\left(\cfrac{x\pi}{L}\right)\end{Bmatrix}$",
            12: r"$u^{*~(%i)}=\begin{Bmatrix}0 & \cfrac{xy(y-L)}{L^3}\end{Bmatrix}$",
            13: r"$u^{*~(%i)}=\begin{Bmatrix}\cfrac{y^2}{L^2}\sin\left(\cfrac{x\pi}{L}\right) & 0\end{Bmatrix}$",
            14: r"$u^{*~(%i)}=\begin{Bmatrix}0 & \cfrac{x^2}{L^2}\sin\left(\cfrac{y\pi}{L}\right)\end{Bmatrix}$",
            15: r"$u^{*~(%i)}=\begin{Bmatrix}\cfrac{xy(x-L)}{L^2}\sin\left(\cfrac{x^2y^2}{L^4}\right) & 0\end{Bmatrix}$",
            16: r"$u^{*~(%i)}=\begin{Bmatrix}\sin\left(\cfrac{x\pi}{L}\right)\cfrac{1}{L} & 0\end{Bmatrix}$",
            17: r"$u^{*~(%i)}=\begin{Bmatrix}0 & \sin\left(\cfrac{y\pi}{L}\right)\cfrac{1}{L}\end{Bmatrix}$",
            18: r"$u^{*~(%i)}=\begin{Bmatrix}\sin\left(\cfrac{x^3\pi}{L^3}\right)\cfrac{1}{L^3} & 0\end{Bmatrix}$",
            19: r"$u^{*~(%i)}=\begin{Bmatrix}0 & \sin\left(\cfrac{y^3\pi}{L^3}\right)\cfrac{1}{L^3}\end{Bmatrix}$",
      
}

n_vfs = len(virtual_disp.keys())

cols = 5
rows = 2

n_figs = math.ceil(n_vfs / (cols*rows))

figs = [plt.figure(num=i, figsize=(19.20,10.8)) for i in range(n_figs)]
[fig.subplots_adjust(wspace=0.35, hspace=0.25) for fig in figs]

specs = [gridspec.GridSpec(ncols=cols, nrows=rows, figure=fig) for fig in figs]

# Add axes to figure

for k, fig in enumerate(figs):
    
    k_row = 0
    k_col = 0

    for i in range(n_vfs):

        if k_col == cols:
            if i > ((cols*rows) - 1):
                k_row = 0
            else:
                k_row += 1
            
            k_col = 0

        fig.add_subplot(specs[k][k_row,k_col])

        k_col +=1

j = 0
#norm = TwoSlopeNorm(vmin=-0.1, vcenter=0, vmax=0.1)
for i, (key, disp) in enumerate(virtual_disp.items()):
    
    k = i % (cols*rows)

    if i >= (cols*rows) and i % (cols*rows) == 0:
        j += 1

    z = np.ones((x.shape[0]-1,x.shape[1]-1))
    z[:] = np.nan

    figs[j].axes[k].set_title(formulas[key] % (key), pad=30, fontsize=10)
    figs[j].axes[k].set_box_aspect(1)
    
    # Draw displaced nodes and mesh
    new_x = x + disp[0]
    new_y = y + disp[1]
   
    figs[j].axes[k].pcolormesh(new_x, new_y, np.sqrt(disp[0]**2 + disp[1]**2), edgecolors='k', linewidths=0.2, antialiased=True, clip_on=False, snap=True, cmap=plt.cm.get_cmap('coolwarm'))
    
    k += 1

# for i, fig in enumerate(figs):
#     plt.savefig("virtual_fields_%i.png" % (i), format="png", dpi=300)
#     plt.close()
# #plt.show()

from matplotlib.backends.backend_pdf import PdfPages

pp = PdfPages('virtual_fields.pdf')
for fig in figs:
    pp.savefig(fig)
pp.close()

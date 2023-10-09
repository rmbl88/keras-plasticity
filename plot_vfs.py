from re import L
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
coords = np.linspace(0, int(LENGTH), 100)

x, y = np.meshgrid(coords, coords)

#nodes =  np.vstack(map(np.ravel, np.meshgrid(coords, coords))).T

#n_nodes = nodes.shape[0]
        
#x = nodes[:,0]        
#y = nodes[:,1]

zeros_ = np.zeros_like(x)
pi_=math.pi
virtual_disp = {
            1: [zeros_, -(x**2-LENGTH)*y/LENGTH**2],
            2: [np.sin(2*math.pi*x/(3*LENGTH)),zeros_],
            3: [(math.pi*y**2/LENGTH**2)*np.sin(2*math.pi*x/LENGTH),(-2*y/LENGTH**2)*np.sin(math.pi*x/LENGTH)**2],
            4: [y*np.sin(x*math.pi/LENGTH)/LENGTH, x*np.sin(y*math.pi/LENGTH**2)/LENGTH],
            5: [zeros_,np.sin(y*math.pi/LENGTH)*np.sin(x*math.pi/LENGTH)]
            #5: [(2*math.pi*y/LENGTH**2)*np.sin(2*math.pi*x/LENGTH),(-y/LENGTH)*np.sin(math.pi*x/LENGTH)**2],
            #6: [np.square(y)*np.sin(x*math.pi/LENGTH)/LENGTH**2, np.square(x)*np.sin(y*math.pi/LENGTH)/LENGTH**2]

            ######
            # 1: [x/LENGTH, zeros_],
            # 2: [zeros_,np.sin(2*y*math.pi/(3*LENGTH))],
            # 3: [zeros_,y*np.sin(x*math.pi/(LENGTH))**2/LENGTH],
            # 4: [np.sin(2*math.pi*x/(3*LENGTH)),zeros_],
            # 5: [(math.pi*y**2/LENGTH**2)*np.sin(2*math.pi*x/LENGTH),(-2*y/LENGTH)*np.sin(math.pi*x/LENGTH)**2],
            # 6: [np.square(y)*np.sin(x*math.pi/LENGTH)/LENGTH**2,np.square(x)*np.sin(y*math.pi/LENGTH)/LENGTH**2],
            # 7: [y**2*np.cos((math.pi+x)/(2*LENGTH))/LENGTH**2,x**2*np.cos((math.pi+y)/(2*LENGTH))/LENGTH**2]
            ######
            # 1: [x/LENGTH,zeros_],
            # 2: [np.sin(2*math.pi*x/(3*LENGTH)),zeros_],
            # 3: [(math.pi*y**2/LENGTH**2)*np.sin(2*math.pi*x/LENGTH),(-2*y/LENGTH)*np.sin(math.pi*x/LENGTH)**2],
            # 4: [np.square(y)*np.sin(x*math.pi/LENGTH)/LENGTH**2, np.square(x)*np.sin(y*math.pi/LENGTH)/LENGTH**2],
            # 2: [zeros_, y/LENGTH],
            # 3: [zeros_, y*(np.square(x)-x*LENGTH)/LENGTH**2],
            # 4: [zeros_, np.sin(x*math.pi/LENGTH)*np.sin(y*math.pi/LENGTH)],
            # 5: [x*y*(x-LENGTH)/LENGTH**2, y*x*(y-LENGTH)/LENGTH**2],
            # 6: [y**2*np.sin(x*math.pi/LENGTH)/LENGTH**2, x**2*np.sin(y*math.pi/LENGTH)/LENGTH**2],
            # 7: [np.sin(y*math.pi/LENGTH) * np.sin(x*math.pi/LENGTH), zeros_],
            # #6: [x*y*(x-LENGTH)/LENGTH**2,zeros_],
            # 8: [np.square(x)*(LENGTH-x)*np.sin(math.pi*y/LENGTH)/LENGTH**3,zeros_],
            # 9: [zeros_, (LENGTH**3-x**3)*np.sin(math.pi*y/LENGTH)/LENGTH**3],
            # 10: [(x*LENGTH**2-x**3)*np.sin(math.pi*y/LENGTH)/LENGTH**3,zeros_],
            # 11: [(x*y*(x-LENGTH)/LENGTH**2)*np.sin(y*math.pi/LENGTH), zeros_],
            # 12: [zeros_, (x*y*(y-LENGTH)/LENGTH**2)*np.sin(x*math.pi/LENGTH)],
            # # 12: [zeros_,x*y*(y-LENGTH)/LENGTH**2],
            # 13: [y**2*np.sin(x*math.pi/LENGTH)/LENGTH**2, zeros_],
            # 14: [zeros_, x**2*np.sin(y*math.pi/LENGTH)/LENGTH**2],
            # 15: [(x*y*(x-LENGTH)/LENGTH**2)*np.sin(x**2*y**2/LENGTH**4), zeros_],
            # 16: [np.sin(x*math.pi/LENGTH), zeros_],
            # 17: [zeros_, np.sin(y*math.pi/LENGTH)],
            # 18: [np.sin(x**3*math.pi/LENGTH**3), zeros_],
            # 19: [zeros_, np.sin(y**3*math.pi/LENGTH**3)],
                   
        }

formulas = {
            1: r"$u^{*~(%i)}=\begin{Bmatrix}\cfrac{x}{L} & 0\end{Bmatrix}$",
            2: r"$u^{*~(%i)}=\begin{Bmatrix}0 & \sin\left(\cfrac{2\pi{}y}{3L}\right)\end{Bmatrix}$",
            3: r"$u^{*~(%i)}=\begin{Bmatrix}0 & \cfrac{y}{L}\sin^{2}\left(\cfrac{\pi{}x}{L}\right)\end{Bmatrix}$",
            4: r"$u^{*~(%i)}=\begin{Bmatrix}\sin\left(\cfrac{2\pi{}x}{3L}\right) & 0\end{Bmatrix}$",
            5: r"$u^{*~(%i)}=\begin{Bmatrix}\cfrac{\pi{}y^2}{L^2}\sin\left(\cfrac{2\pi{}x}{L}\right) & -\cfrac{2y}{L}\sin^{2}\left(\cfrac{\pi{}x}{L}\right)\end{Bmatrix}$",
            #6: r"$u^{*~(%i)}=\begin{Bmatrix}\cfrac{y^{2}}{L^2}\sin\left(\cfrac{\pi{}x}{L}\right) & \cfrac{x^{2}}{L^2}\sin\left(\cfrac{\pi{}y}{L}\right)\end{Bmatrix}$",
            #7: r"$u^{*~(%i)}=\begin{Bmatrix}\cfrac{y^{2}}{L^2}\sin\left(\cfrac{\pi{}x}{L}\right) & \cfrac{x^{2}}{L^2}\sin\left(\cfrac{\pi{}y}{L}\right)\end{Bmatrix}$"
            # 1: r"$u^{*~(%i)}=\begin{Bmatrix}\cfrac{x}{L} & 0\end{Bmatrix}$",
            # 2: r"$u^{*~(%i)}=\begin{Bmatrix}0 & \cfrac{y}{L}\end{Bmatrix}$",
            # 3: r"$u^{*~(%i)}=\begin{Bmatrix}0 & y\cfrac{x^2-xL}{L^2}\end{Bmatrix}$",
            # 4: r"$u^{*~(%i)}=\begin{Bmatrix}0 & \sin\left(\cfrac{x\pi}{L}\right)\sin\left(\cfrac{y\pi}{L}\right)\end{Bmatrix}$",
            # 5: r"$u^{*~(%i)}=\begin{Bmatrix}\cfrac{xy(x-L)}{L^2} & \cfrac{xy(y-L)}{L^2}\end{Bmatrix}$",
            # 6: r"$u^{*~(%i)}=\begin{Bmatrix}\cfrac{y^2}{L^2}\sin\left(\cfrac{x\pi}{L}\right) & \cfrac{x^2}{L^2}\sin\left(\cfrac{y\pi}{L}\right)\end{Bmatrix}$",
            # 7: r"$u^{*~(%i)}=\begin{Bmatrix}\sin\left(\cfrac{x\pi}{L}\right)\sin\left(\cfrac{y\pi}{L}\right) & 0\end{Bmatrix}$",
            # # 6: r"$u^{*~(%i)}=\begin{Bmatrix}\cfrac{xy(x-L)}{L^2} & 0\end{Bmatrix}$",
            # 8: r"$u^{*~(%i)}=\begin{Bmatrix}\cfrac{x^2(L-x)}{L^3}\sin\left(\cfrac{y\pi}{L}\right) & 0\end{Bmatrix}$",
            # 9: r"$u^{*~(%i)}=\begin{Bmatrix} 0 & \cfrac{L^3-x^3}{L^3}\sin\left(\cfrac{y\pi}{L}\right)\end{Bmatrix}$",
            # 10: r"$u^{*~(%i)}=\begin{Bmatrix}\cfrac{xL^2-x^3}{L^3}\sin\left(\cfrac{y\pi}{L}\right) & 0\end{Bmatrix}$",
            # 11: r"$u^{*~(%i)}=\begin{Bmatrix}\cfrac{xy(x-L)}{L^2}\sin\left(\cfrac{y\pi}{L}\right) & 0\end{Bmatrix}$",
            # 12: r"$u^{*~(%i)}=\begin{Bmatrix}0 & \cfrac{xy(y-L)}{L^2}\sin\left(\cfrac{x\pi}{L}\right)\end{Bmatrix}$",
            # # 12: r"$u^{*~(%i)}=\begin{Bmatrix}0 & \cfrac{xy(y-L)}{L^2}\end{Bmatrix}$",
            # 13: r"$u^{*~(%i)}=\begin{Bmatrix}\cfrac{y^2}{L^2}\sin\left(\cfrac{x\pi}{L}\right) & 0\end{Bmatrix}$",
            # 14: r"$u^{*~(%i)}=\begin{Bmatrix}0 & \cfrac{x^2}{L^2}\sin\left(\cfrac{y\pi}{L}\right)\end{Bmatrix}$",
            # 15: r"$u^{*~(%i)}=\begin{Bmatrix}\cfrac{xy(x-L)}{L^2}\sin\left(\cfrac{x^2y^2}{L^4}\right) & 0\end{Bmatrix}$",
            # 16: r"$u^{*~(%i)}=\begin{Bmatrix}\sin\left(\cfrac{x\pi}{L}\right) & 0\end{Bmatrix}$",
            # 17: r"$u^{*~(%i)}=\begin{Bmatrix}0 & \sin\left(\cfrac{y\pi}{L}\right)\end{Bmatrix}$",
            # 18: r"$u^{*~(%i)}=\begin{Bmatrix}\sin\left(\cfrac{x^3\pi}{L^3}\right) & 0\end{Bmatrix}$",
            # 19: r"$u^{*~(%i)}=\begin{Bmatrix}0 & \sin\left(\cfrac{y^3\pi}{L^3}\right)\end{Bmatrix}$",
      
}

n_vfs = len(virtual_disp.keys())

cols = 4
rows = 2

n_figs = math.ceil(n_vfs / (cols*rows))

#figs = [plt.figure(num=i, figsize=(15,15)) for i in range(n_vfs)]

figs = [plt.figure(num=i, figsize=(19.2,10.8)) for i in range(n_figs)]
[fig.subplots_adjust(wspace=0.2, hspace=0.25) for fig in figs]

specs = [gridspec.GridSpec(ncols=cols, nrows=rows, figure=fig) for fig in figs]

# Add axes to figure

# for k, fig in enumerate(figs):
#     fig.add_subplot(111)

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

    figs[j].axes[k].set_title(formulas[key] % (key), pad=30, fontsize=15)
    figs[j].axes[k].set_box_aspect(1)

    # figs[i].axes[0].set_title(formulas[key] % (key), pad=75, fontsize=60)
    # figs[i].axes[0].set_box_aspect(1)
    
    # Draw displaced nodes and mesh
    new_x = x + disp[0]
    new_y = y + disp[1]
   

    Z = np.sqrt(disp[0]**2 + disp[1]**2)
    #pcol = figs[j].axes[k].pcolormesh(new_x, new_y, np.sqrt(disp[0]**2 + disp[1]**2), linewidths=0, antialiased=True, rasterized=True, clip_on=False, cmap=plt.cm.get_cmap('coolwarm'))
    figs[j].axes[k].pcolor(new_x, new_y, Z, rasterized=True, cmap=plt.cm.get_cmap('gist_yarg'))
    #pcol.set_edgecolor('face')
    #figs[j].axes[k].contourf(new_x, new_y, Z, 500, cmap=plt.cm.get_cmap('coolwarm'), antialiased=True)
    #figs[j].axes[k].imshow(Z, origin='lower', cmap=plt.cm.get_cmap('coolwarm'), interpolation='bilinear')
    

    k += 1

# for i, fig in enumerate(figs):
#     plt.savefig("virtual_fields_%i.pdf" % (i), format="pdf", dpi=300, bbox_inches='tight')
#     plt.close()
plt.show()

# from matplotlib.backends.backend_pdf import PdfPages

# pp = PdfPages('virtual_fields.pdf')
# for fig in figs:
#     pp.savefig(fig)
# pp.close()

import numpy as np
from numpy.core.numeric import empty_like, zeros_like
from tensorflow.python.keras.backend import zeros
from torch.functional import meshgrid
from constants import *
import matplotlib.pyplot as plt
import math
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm, ListedColormap


plt.rcParams.update(PARAMS)

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
            # 9: [zeros_, (LENGTH ** 3 - x ** 3) * np.sin(y * math.pi / LENGTH) / (LENGTH ** 3)],
            # 10: [(x * LENGTH **2 - x ** 3) * np.sin(math.pi * y / LENGTH) / (LENGTH ** 3), zeros_]
            #11: [zeros_, x * y * (y - LENGTH) / LENGTH ** 3]
            
        }

n_vfs = len(virtual_disp.keys())

cols = 3
rows = math.ceil(n_vfs / cols)

fig = plt.figure(tight_layout=True)
fig.subplots_adjust(wspace=0.35, hspace=0.35)
spec = gridspec.GridSpec(ncols=cols, nrows=rows, figure=fig)



# Add axes to figure
k_row = 0
k_col = 0
for i in range(n_vfs):

    if k_col > 2:
        k_row += 1
        k_col = 0

    fig.add_subplot(spec[k_row,k_col])

    k_col +=1


#norm = TwoSlopeNorm(vmin=-0.1, vcenter=0, vmax=0.1)
for i, (key, disp) in enumerate(virtual_disp.items()):
    z = np.ones_like(x)
    fig.axes[i].set_title(r'$u^{*~(%i)}$' % key)
    fig.axes[i].axis('off')
    fig.axes[i].set_box_aspect(1)
    #fig.axes[i].set_aspect('equal')
    # Draw underlying nodes and mesh
    #fig.axes[i].plot(x,y, color='lightgray', marker='.', clip_on=False, markersize=1.5)
    fig.axes[i].pcolormesh(x, y, z, edgecolors='r',  linewidths=1, antialiased=True, clip_on=False, cmap=ListedColormap(['white']))
    # fig.axes[i].vlines(x[0], *y[[0,-1],0], 'k', clip_on=False, lw=0.)
    # fig.axes[i].hlines(y[:,0], *x[0, [0,-1]], 'k', clip_on=False, lw=0.3)
    # Draw displaced nodes and mesh
    new_x = x + disp[0]
    new_y = y + disp[1]
    fig.axes[i].plot(x + disp[0], y + disp[1], '.k', clip_on=False, markersize=3)
  
    fig.axes[i].pcolormesh(new_x, new_y, z, edgecolors='k', linewidths=0.85, antialiased=True, clip_on=False, cmap=ListedColormap(['white']), alpha=0.65)

plt.show()



# f2_ax1 = fig2.add_subplot(spec2[0, 0])
# f2_ax2 = fig2.add_subplot(spec2[0, 1])
# f2_ax3 = fig2.add_subplot(spec2[1, 0])
# f2_ax4 = fig2.add_subplot(spec2[1, 1])

# x, y = np.meshgrid(coords, coords)
# plt.plot(x, y, ".k")
# plt.vlines(x[0], *y[[0,-1],0])
# plt.hlines(y[:,0], *x[0, [0,-1]])
# plt.axis('off')
# plt.show()
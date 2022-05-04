from turtle import color
from matplotlib.pyplot import draw
import numpy as np
import torch
import joblib
import matplotlib.animation as animation
import pylab as plt
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

def animate(i):
    global contours, v_u, fig, cmap, bar_x, bar_y

    for j,contour in enumerate(contours):

        if j == 0:
            u_ = torch.reshape(v_u[vf][i][::2],X.shape)
        else:
            u_ = torch.reshape(v_u[vf][i][1::2],X.shape)

        v_min = torch.min(u_)
        v_max = torch.max(u_)
        bounds = np.linspace(v_min*1.1, v_max*1.1,13)
        norm = colors.BoundaryNorm(bounds,cmap.N)

        contour.set_data(u_)
        contour.set_norm(norm)
        

        # if j == 0:
        #     bar_x.set_ticks(bounds)
               
        # else:
        #     bar_y.set_ticks(bounds)
            
        
        contour.axes.set_title('t_inc = %i' % (i))

    return contours

x = np.linspace(0,3,4)
y = np.linspace(0,3,4)

X,Y = np.meshgrid(x,y)

vf_u = joblib.load('sbvfs.pkl')

tags = list(vf_u.keys())

for tag in tags:

    v_u = torch.stack([v for k,v in vf_u[tag].items() if v is not None],1)

    cmap = plt.cm.get_cmap('jet')    

    for vf in range(v_u.shape[0]):

        fig, [ax1, ax2] = plt.subplots(1,2)
        fig.suptitle(tag+'_VFu_'+str(vf))
        fig.set_size_inches(16, 8, forward=True)
        ax1.set_xlim(0,3)
        ax1.set_ylim(0,3)
        ax2.set_xlim(0,3)
        ax2.set_ylim(0,3)
        ax1.set_xlabel('u*_x')
        ax2.set_xlabel('u*_y')
        #cont_x = ax1.contourf(X, Y, torch.reshape(v_u[vf][0][::2],X.shape), 21)
        cont_x=ax1.imshow(np.zeros_like(X), origin='lower', interpolation='bicubic', aspect='equal',extent=[0, 3, 0, 3], cmap=cmap)
        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        divider_x = make_axes_locatable(ax1)
        cax_x = divider_x.append_axes("right", size="5%", pad=0.05)
        bar_x = plt.colorbar(cont_x,cax=cax_x)
        cont_y=ax2.imshow(np.zeros_like(X), origin='lower', interpolation='bicubic', aspect='equal',extent=[0, 3, 0, 3], cmap=cmap)
        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        divider_x = make_axes_locatable(ax2)
        cax_y = divider_x.append_axes("right", size="5%", pad=0.05)
        bar_x = plt.colorbar(cont_y,cax=cax_y)
        bar_y = plt.colorbar(cont_y,cax=cax_y)
        contours = [cont_x, cont_y]
        anim = animation.FuncAnimation(fig, animate,frames=v_u.shape[1], repeat=False)
        anim.save('animation_%i.gif'%(vf), writer=animation.PillowWriter(fps=0.5))
        plt.close(fig)
        print('hey')


print('hey')
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
    global contours, v_u, fig

    for j,contour in enumerate(contours):

        if j == 0:
            u_ = torch.reshape(v_u[vf][i][::2],X.shape)
        else:
            u_ = torch.reshape(v_u[vf][i][1::2],X.shape)

        v_min = torch.min(u_)
        v_max = torch.max(u_)
        contour.set_data(u_)
        contour.set_clim(v_min, v_max)
        contour.axes.set_title('t_inc = %i' % (i))
        fig.canvas.draw()
        fig.canvas.flush_events()
        
    
    
    #cont_x=plt.imshow(u_x, origin='lower', interpolation='bicubic', aspect='auto',extent=[0, 3, 0, 3],animated=True)
    #cont_y=plt.imshow(u_y, origin='lower', interpolation='bicubic', aspect='auto',extent=[0, 3, 0, 3],animated=True)
    #cont_x.axes.set_title('t_inc = %i' % (i))
    #bar_x = plt.colorbar(cont_x,ax=cont_x.axes)
    #cont_y.axes.set_title('t_inc = %i' % (i))
    #bar_y = plt.colorbar(cont_y,ax=cont_y.axes)

    return contours

x = np.linspace(0,3,4)
y = np.linspace(0,3,4)

X,Y = np.meshgrid(x,y)

vf_u = joblib.load('sbvfs.pkl')

tags = list(vf_u.keys())

for tag in tags:

    v_u = torch.stack([v for k,v in vf_u[tag].items() if v is not None],1)
    
    u_x_min = torch.min(v_u[:,:,::2])
    u_x_max = torch.max(v_u[:,:,::2])
    
    u_y_min = torch.min(v_u[:,:,1::2])
    u_y_max = torch.max(v_u[:,:,1::2])

    levels_x = np.linspace(u_x_min,u_x_max,5)
    levels_y = np.linspace(u_y_min,u_y_max,5)

    cmap = plt.cm.jet

    norm_x = colors.BoundaryNorm(levels_x, cmap.N)
    norm_y = colors.BoundaryNorm(levels_y, cmap.N)
    

    for vf in range(v_u.shape[0]):

        fig, [ax1, ax2] = plt.subplots(1,2)
        fig.suptitle(tag+'_VFu_'+str(vf))
        fig.set_size_inches(12, 8, forward=True)
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

        print('hey')


print('hey')
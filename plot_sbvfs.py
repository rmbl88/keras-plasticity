from cProfile import label
from turtle import color
from matplotlib.pyplot import bar, draw
import numpy as np
import torch
import joblib
import matplotlib.animation as animation
import pylab as plt
from matplotlib import colors, ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec
import math


def animate(i):
    
    global contours, v_u, v_e, fig, cmap, bar_x, bar_y, bar_exx, bar_eyy, bar_exy
    
    cbs=[bar_x,bar_y,bar_exx,bar_eyy,bar_exy]
    
    k = 0
    for j,contour in enumerate(contours):

        if j == 0:
            u_ = torch.reshape(v_u[vf][i][::2],X.shape)
            contour.axes.set_title('t_inc = %i' % (i))
        elif j == 1:
            u_ = torch.reshape(v_u[vf][i][1::2],X.shape)
            contour.axes.set_title('t_inc = %i' % (i))
        else:
            u_ = torch.reshape(v_e[vf][i][:,k],XC.shape)
            k += 1
        
        contour.set_data(u_)
       
        v_min = torch.min(u_)
        v_max = torch.max(u_)

        if v_min != v_max:
            bounds = np.linspace(v_min*1.1, v_max*1.1,13)
            norm = colors.BoundaryNorm(bounds,cmap.N)
            contour.set_norm(norm)            
            cbs[j].set_ticks(bounds)
            cbs[j].set_ticklabels(bounds)
        cbs[j].ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3e'))
        

    return contours

def animate_2(i):

    global contours, cbars, error

    for j,contour in enumerate(contours):
        
        w = torch.reshape(error[i],XC.shape)
        contour.set_data(w)
        contour.axes.set_title('t_inc = %i' % (i))

        v_min = torch.min(w)
        v_max = torch.max(w)

        if v_min != v_max:
            bounds = np.linspace(v_min*1.1, v_max*1.1,13)
            norm = colors.BoundaryNorm(bounds,cmap.N)
            contour.set_norm(norm)            
            cbars[j].set_ticks(bounds)
            cbars[j].set_ticklabels(bounds)
        cbars[j].ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))

    return contours

x = np.linspace(0,3,4)
y = np.linspace(0,3,4)

X,Y = np.meshgrid(x,y)

xc = np.linspace(0.5,2.5,3)
yc = np.linspace(0.5,2.5,3)

XC, YC = np.meshgrid(xc,yc)

#vfs, w_virt = joblib.load('sbvfs_m80b80.pkl')

FOLDER = '9-elem-50-plastic_sbvf_abs/'

vfs = np.load('outputs/' + FOLDER + 'logs/sbvfs.npy', allow_pickle=True).flat[0]
w_virt = np.load('outputs/' + FOLDER + 'logs/w_virt.npy', allow_pickle=True).flat[0]

tags = list(vfs.keys())

cmap = plt.cm.get_cmap('jet')

# k=0
# for tag in tags:
#     v_u = torch.stack([torch.from_numpy(v) for k,v in vfs[tag]['u'].items() if v is not None],1)
#     v_e = torch.stack([torch.from_numpy(v) for k,v in vfs[tag]['e'].items() if v is not None],1)

#     for vf in range(v_u.shape[0]):
#         print('\rProcessing animation %i of %i' % (k+1,len(tags)*len(range(v_u.shape[0]))), end='')
#         #fig, [ax1, ax2] = plt.subplots(1,2)
#         fig = plt.figure()
#         fig.suptitle(tag+'_VFu_'+str(vf))
#         fig.set_size_inches(18, 9, forward=True)
#         fig.subplots_adjust(wspace=0.4)
#         fig.tight_layout()
#         gs = GridSpec(2, 6, figure=fig)

#         ax1 = fig.add_subplot(gs[0,:3])
#         ax1.set_xlim(0,3)
#         ax1.set_ylim(0,3)
#         ax1.set_xlabel('u*_x')

#         ax2 = fig.add_subplot(gs[0,3:])
#         ax2.set_xlim(0,3)
#         ax2.set_ylim(0,3)
#         ax2.set_xlabel('u*_y')

#         ax4 = fig.add_subplot(gs[1,:2])
#         ax4.set_xlabel('e*_xx')
#         ax5 = fig.add_subplot(gs[1,2:4])
#         ax5.set_xlabel('e*_yy')
#         ax6 = fig.add_subplot(gs[1,4:])
#         ax6.set_xlabel('e*_xy')

#         #cont_x = ax1.contourf(X, Y, torch.reshape(v_u[vf][0][::2],X.shape), 21)
#         cont_x=ax1.imshow(np.zeros_like(X), origin='lower', interpolation='bicubic', aspect='equal',extent=[0, 3, 0, 3], cmap=cmap)
#         # create an axes on the right side of ax. The width of cax will be 5%
#         # of ax and the padding between cax and ax will be fixed at 0.05 inch.
#         divider_x = make_axes_locatable(ax1)
#         cax_x = divider_x.append_axes("right", size="5%", pad=0.05)
#         bar_x = plt.colorbar(cont_x,cax=cax_x,format='%.3e') 
       
#         cont_y=ax2.imshow(np.zeros_like(X), origin='lower', interpolation='bicubic', aspect='equal',extent=[0, 3, 0, 3], cmap=cmap)
#         # create an axes on the right side of ax. The width of cax will be 5%
#         # of ax and the padding between cax and ax will be fixed at 0.05 inch.
#         divider_y = make_axes_locatable(ax2)
#         cax_y = divider_y.append_axes("right", size="5%", pad=0.05)
#         bar_y = plt.colorbar(cont_y,cax=cax_y,format='%.3e')        

#         cont_exx=ax4.imshow(np.zeros_like(XC), origin='lower', interpolation='bicubic', aspect='equal',extent=[0, 3, 0, 3], cmap=cmap)
#         # create an axes on the right side of ax. The width of cax will be 5%
#         # of ax and the padding between cax and ax will be fixed at 0.05 inch.
#         divider_exx = make_axes_locatable(ax4)
#         cax_exx = divider_exx.append_axes("right", size="5%", pad=0.05)
#         bar_exx = plt.colorbar(cont_exx,cax=cax_exx,format='%.3e')

#         cont_eyy=ax5.imshow(np.zeros_like(XC), origin='lower', interpolation='bicubic', aspect='equal',extent=[0, 3, 0, 3], cmap=cmap)
#         # create an axes on the right side of ax. The width of cax will be 5%
#         # of ax and the padding between cax and ax will be fixed at 0.05 inch.
#         divider_eyy = make_axes_locatable(ax5)
#         cax_eyy = divider_eyy.append_axes("right", size="5%", pad=0.05)
#         bar_eyy = plt.colorbar(cont_eyy,cax=cax_eyy,format='%.3e')

#         cont_exy=ax6.imshow(np.zeros_like(XC), origin='lower', interpolation='bicubic', aspect='equal',extent=[0, 3, 0, 3], cmap=cmap)
#         # create an axes on the right side of ax. The width of cax will be 5%
#         # of ax and the padding between cax and ax will be fixed at 0.05 inch.
#         divider_exy = make_axes_locatable(ax6)
#         cax_exy = divider_exy.append_axes("right", size="5%", pad=0.05)
#         bar_exy = plt.colorbar(cont_exy,cax=cax_exy,format='%.3e')
        
#         contours = [cont_x, cont_y, cont_exx, cont_eyy, cont_exy]

#         anim = animation.FuncAnimation(fig, animate,frames=v_u.shape[1], repeat=False)
#         anim.save('anim_%s_vf_%i.gif'%(tag,vf), writer=animation.PillowWriter(fps=1))
#         plt.close(fig)
#         k+=1
        
tags = list(w_virt.keys())
for i,tag in enumerate(tags):
    print('\rProcessing stats plot %i of %i' % (i+1,len(tags)), end='')

    w_int = np.array(list(w_virt[tag]['w_int'].values()))
    w_int_real = np.array(list(w_virt[tag]['w_int_real'].values()))
    w_ext = np.array(list(w_virt[tag]['w_ext'].values()))
    w_int_real[0] = 1e-16
    # w_int = np.sum(w_int,-1,keepdim=True)
    # w_int_real = np.sum(w_int_real,-1,keepdim=True)
    # ivw_sort = torch.sort(torch.abs(w_int_real.detach()).flatten(),descending=True).values
    # numSteps = math.floor(0.3*len(ivw_sort))
    # alpha = torch.mean(ivw_sort[0:numSteps]) * torch.ones((w_int_real.shape[1],1))
    w_int_err = np.abs(w_int-w_int_real)*100/w_int_real
    t_steps = np.array(list(w_virt[tag]['w_int'].keys()))

    fig = plt.figure()
    fig.suptitle('%s' % (tag))
    fig.set_size_inches(16, 8, forward=True)
    fig.subplots_adjust(wspace=0.6)
    fig.tight_layout()
    g_s = GridSpec(1, 6, figure=fig)

    ax1 = fig.add_subplot(g_s[0,0:3])
    #ax.plot(torch.sum(w_int_real,0))
    #ax.plot(torch.sum(w_ext,0))
    ax1.plot(w_int, label='W_int (ANN)')
    ax1.plot(w_int_real, label='W_int (Abaqus)')
    ax1.plot(w_ext, label='W_ext (Abaqus)')
    
    ax1.set_ylabel('Virtual work [J]')
    ax1.set_xlabel('t_step')
    ax1.legend(loc='best')

    ax2 = fig.add_subplot(g_s[0,3:])
    
    ax2.bar(t_steps[1:],w_int_err[1:])
    ax2.set_ylabel('W_int rel. error [%]')
    ax2.set_xlabel('t_step')
    
    #plt.show()
    plt.savefig('stats_%s.png'%(tag), format="png", dpi=100, bbox_inches='tight')
    plt.close(fig)
    
        
print('hey')
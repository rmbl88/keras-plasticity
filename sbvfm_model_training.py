# ---------------------------------
#    Library and function imports
# ---------------------------------
from pyexpat import model
from re import L
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
from scipy.signal import find_peaks_cwt
from scipy.signal import argrelmax, argrelmin
import matplotlib.pyplot as plt
import cProfile, pstats
import os
import shutil
import joblib
from sklearn import model_selection
from constants import *
from functions import (
    SBVFLoss,
    layer_wise_lr,
    sbvf_loss,
    draw_graph,
    load_dataframes,
    prescribe_u,
    select_features_multi,
    standardize_data,
    plot_history,
    read_mesh,
    global_strain_disp,
    param_deltas,
    global_dof,
    stress_from_cholesky)
from functions import (
    weightConstraint,
    EarlyStopping,
    NeuralNetwork,
    Element)
from functools import partial
import copy
from torch import batch_norm, nn
import tensorflow as tf
import pandas as pd
import random
import numpy as np
import math
import torch
import time
import itertools
import wandb
from torch.autograd.functional import jacobian
import time

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.ticker import MaxNLocator
import multiprocessing
import time
import random
from tkinter import *
import geotorch
from warmup_scheduler import GradualWarmupScheduler
from ignite.handlers import create_lr_scheduler_with_warmup
from ignite.engine import *
from multiprocessing import Pool

# -----------------------------------------
#   DEPRECATED IMPORTS
# -----------------------------------------
# from torch.nn.utils import (
#   parameters_to_vector as Params2Vec,
#   vector_to_parameters as Vec2Params
# )

# ----------------------------------------
#        Class definitions
# ----------------------------------------
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, deformation, stress, force, coord, info, list_IDs, batch_size, shuffle=False, std=True, t_pts=1, scaler=None):
        super().__init__()

        self.X = deformation.iloc[list_IDs].reset_index(drop=True)
        self.y = stress.iloc[list_IDs].reset_index(drop=True)
        self.f = force.iloc[list_IDs].reset_index(drop=True)
        self.coord = coord[['dir','id','cent_x','cent_y','area']].iloc[list_IDs].reset_index(drop=True)
        self.tag = info['tag'].iloc[list_IDs].reset_index(drop=True)
        #self.t = info[['inc','t','exx_p_dot','eyy_p_dot','exy_p_dot']].iloc[list_IDs].reset_index(drop=True)
        self.t = info[['inc','t','d_exx','d_eyy','d_exy']].iloc[list_IDs].reset_index(drop=True)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.list_IDs = list_IDs
        self.std = std
        self.t_pts = t_pts
        self.scaler_x = scaler

        if self.std == True:
            self.standardize()

        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = np.array([self.indexes[index]+i for i in range(self.batch_size)])  # Generate indexes of the batch

        # Generate data according to batch size specifications
        if self.shuffle == True:
            index_groups = np.array_split(indexes, self.t_pts)
            #permuted = [np.random.permutation(index_groups[i]) for i in range(len(index_groups))]
            permuted = np.random.permutation(index_groups)
            indexes = np.hstack(permuted)

        x, y, f, coord, tag, t = self.__data_generation(indexes)

        return x, y, f, coord, tag, t

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(0, len(self.list_IDs), self.batch_size)
        #if self.shuffle == True:
        np.random.shuffle(self.indexes)

    def standardize(self):
        'Standardizes neural network input data'
        idx = self.X.index
        # if self.scaler_x != None:
        #     self.X, _, _, _, _, _ = standardize_data(self.X, self.y, self.f, scaler_x=self.scaler_x)
        #else:
        self.X, _, _, self.scaler_x, _, _ = standardize_data(self.X, self.y, self.f)

        self.X = pd.DataFrame(self.X, index=idx)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        x = np.asarray(self.X.iloc[list_IDs_temp], dtype=np.float64)
        y = np.asarray(self.y.iloc[list_IDs_temp], dtype=np.float64)
        f = np.asarray(self.f.iloc[list_IDs_temp], dtype=np.float64)
        coord = np.asarray(self.coord.iloc[list_IDs_temp], dtype=np.float64)
        tag = self.tag.iloc[list_IDs_temp]
        t = self.t.iloc[list_IDs_temp]

        return x, y, f, coord, tag, t

# -------------------------------
#       Plotting classes
# -------------------------------
def plot():    #Function to create the base plot, make sure to make global the lines, axes, canvas and any part that you would want to update later
    def get_plot(ax,style):
        return ax.plot([], style, animated=True)[0]

    global w_ext,w_int,w_int_real,t_loss,v_loss,s_loss,s_err,ax,ax_2,ax_3,canvas

    fig = matplotlib.figure.Figure(figsize=(16, 5),tight_layout=True)

    ax = fig.add_subplot(1,3,1)
    ax_2 = fig.add_subplot(1,3,2)
    ax_3 = fig.add_subplot(1,3,3)
    
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()
    canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)
    canvas._tkcanvas.pack(side=TOP, fill=BOTH, expand=1)

    w_ext, = ax.plot([],label='w_ext')
    w_int, = ax.plot([],label='w_int_ann')
    w_int_real, = ax.plot([],label='w_int')
    ax.set_title(' ')
    ax.set_xlabel('t_pts')
    ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    
    ax.legend(loc='best')

    v_loss, = ax_2.plot([],label='test_loss')
    t_loss, = ax_2.plot([],label='train_loss')
    
    s_err, = ax_3.plot([],label='mse_stress')
    s_loss, = ax_3.plot([],label='s(0)_loss')

    ax_2.set_title('Training curves')
    ax_3.set_title('Constraints loss')

    for i,ax_ in enumerate([ax,ax_2,ax_3]):
        
        ax_.legend(loc='best')
        if i != 0:
            ax_.set_yscale('log')
            ax_.set_xlim([0, 2])
            
            ax_.set_xlabel('Epochs')
            ax_.set_ylabel('Loss')
            ax_.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        
def updateplot(q):

    try:       #Try to check if there is data in the queue
        result, tag = q.get_nowait()

        if result != 'Q':
            if tag == 'loss':

                d1 = np.append(t_loss.get_ydata(),result[0])
                d2 = np.append(v_loss.get_ydata(),result[1])
                d3 = np.append(s_loss.get_ydata(),result[2])
                d4 = np.append(s_err.get_ydata(),result[3])

                epochs = list(range(len(d1)))

                t_loss.set_xdata(epochs)
                t_loss.set_ydata(d1)

                v_loss.set_xdata(epochs)
                v_loss.set_ydata(d2)

                s_loss.set_xdata(epochs)
                s_loss.set_ydata(d3)

                s_err.set_xdata(epochs)
                s_err.set_ydata(d4)

                lower_2 = min([min(a) for a in [d1,d2]])
                upper_2 = max([max(a) for a in [d1,d2]])

                lower_3 = min([min(a) for a in [d3,d4]])
                upper_3 = max([max(a) for a in [d3,d4]])

                ax_2.set_xlim([0, max(epochs)+5])
                ax_2.set_ylim([lower_2-0.5*abs(lower_2), upper_2+0.1*abs(upper_2)])

                ax_3.set_xlim([0, max(epochs)+5])
                ax_3.set_ylim([lower_3-0.5*abs(lower_3), upper_3+0.1*abs(upper_3)])

                ax_2.draw_artist(ax_2.patch)
                ax_2.draw_artist(t_loss)
                ax_2.draw_artist(v_loss)
                
                ax_3.draw_artist(ax_3.patch)
                ax_3.draw_artist(s_loss)
                ax_3.draw_artist(s_err)

                canvas.update()
                canvas.flush_events()

            else:

                w_int_=np.array(list(result['w_int'].values()))
                w_ext_=np.array(list(result['w_ext'].values()))
                w_int_real_=np.array(list(result['w_int_real'].values()))
                t = np.array(list(result['w_int'].keys()))

                lower = min([min(a) for a in [w_int_,w_ext_,w_int_real_]])
                upper = max([max(a) for a in [w_int_,w_ext_,w_int_real_]])

                ax.set_title(tag)
                w_ext.set_xdata(t)
                w_ext.set_ydata(w_ext_)
                w_int.set_xdata(t)
                w_int.set_ydata(w_int_)
                w_int_real.set_xdata(t)
                w_int_real.set_ydata(w_int_real_)

                ax.set_xlim([min(t), max(t)+10])
                ax.set_ylim([lower-0.2*abs(lower), upper+0.1*abs(upper)])

                ax.draw_artist(ax.patch)
                ax.draw_artist(w_ext)
                ax.draw_artist(w_int)
                ax.draw_artist(w_int_real)
                canvas.draw()
                canvas.flush_events()

            window.after(2,updateplot,q)

    except:
        window.after(2,updateplot,q)


# -------------------------------
#       Method definitions
# -------------------------------

def batch_jacobian(f, x):
    f_sum = lambda x: torch.sum(f(x), axis=0)
    return jacobian(f_sum, x, create_graph=True).permute(1,0,2)

def count_parameters(model):
    exceptions = [k for k, v in model.state_dict().items() if 'activations' in k or 'b_norms' in k or 'parametrizations'in k or 'passthrough' in k or 'bias' in k]
    return sum(p.numel() for k, p in model.state_dict().items() if k not in exceptions)

# # Defined at top-level because nested functions are not importable in multiprcessing.pool
# # Refer to: https://stackoverflow.com/questions/52265120/python-multiprocessing-pool-attributeerror
# def sigma_deltas(param_dict, model, x, s):

#         s_=s.reshape((-1,)+s.shape[2:])
#         x_=x.reshape((-1,)+x.shape[2:])
#         with torch.no_grad():
#             model.load_state_dict(param_dict)
#             d_sigma = (s_ - model(x_).detach())

       
#         ##### INCREMENTAL DELTA STRESS #############
#         # n_vfs = delta_stress.shape[0]
#         # n_elems = delta_stress.shape[1] // T_PTS
#         # d_stress = torch.reshape(delta_stress,[n_vfs,T_PTS,n_elems,3])
#         # if idx != None:
#         #     d_stress = d_stress[:,idx]
#         # d_s = torch.zeros_like(d_stress)
#         # d_s[:,1:] = (d_stress[:,1:] - d_stress[:,:-1]) / 0.02

#         # return torch.reshape(d_s[:,incs],delta_stress.shape)
#         # # ##############################################
#         return torch.reshape(d_sigma,s.shape)

def train(q):
    def get_grad_norm(model):
        total_norm = []
        # for p in model.parameters():
        #     if p.grad is not None:
        #         param_norm = p.grad.data.norm(2)
        #         total_norm += param_norm.item() ** 2
        # total_norm = total_norm ** (1. / 2)
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(p=float("inf"))
                total_norm.append(param_norm)
        return torch.mean(torch.as_tensor(total_norm))

    def init_weights(m):
        '''
        Performs the weight initialization of a neural network

        Parameters
        ----------
        m : NeuralNetwork object
            Neural network model, instance of NeuralNework class
        '''
        if isinstance(m, nn.Linear) and (m.bias != None):
            torch.nn.init.kaiming_normal_(m.weight) #RELU
            #torch.nn.init.xavier_normal(m.weight)
            #torch.nn.init.zeros_(m.bias)
            torch.nn.init.ones_(m.bias)
            #m.bias.data.fill_(0.01)

    # Defined at top-level because nested functions are not importable in multiprcessing.pool
    # Refer to: https://stackoverflow.com/questions/52265120/python-multiprocessing-pool-attributeerror
    def sigma_deltas(param_dict, model, x, d_e, s):
    
        s_=s.reshape((-1,)+s.shape[2:])
        x_=x.reshape((-1,)+x.shape[2:])

        n_tests = s.shape[0]
        n_elems = s.shape[1] // T_PTS

        model.eval()
        with torch.no_grad():
            model.load_state_dict(param_dict)
            pred = stress_from_cholesky(model(x_).detach(), d_e, T_PTS, n_elems, n_tests=n_tests, sbvf_gen=True)
            d_sigma = (s_ - pred)
           
        #d_sigma = torch.reshape(d_sigma,s.shape)

        if INCREMENTAL_VFS:
            ##### INCREMENTAL DELTA STRESS #############
            #n_elems = s.shape[1] // T_PTS
            ds_ = torch.reshape(d_sigma,[d_sigma.shape[0],T_PTS,n_elems,3])
            
            dd_sigma = torch.zeros_like(ds_)
            dd_sigma[:,1:] = (ds_[:,1:] - ds_[:,:-1]) / 0.02

            return dd_sigma.reshape(s.shape)
            # # ##############################################
        else:
            return d_sigma.reshape(s.shape)

    def sbv_fields(d_sigma, b_glob, b_inv, n_elems, bcs, active_dof):

        n_dof = b_glob.shape[-1]

        # Reshaping d_sigma for least-square system
        d_s = torch.reshape(d_sigma,(d_sigma.shape[0], T_PTS, n_elems * d_sigma.shape[-1],1))

        v_u = torch.zeros((d_s.shape[0],d_s.shape[1], n_dof, 1))

        # Computing virtual displacements (all dofs)
        v_u[:, :, active_dof] = torch.matmul(b_inv,d_s)

        # Prescribing displacements
        v_u, v_disp = prescribe_u(v_u, bcs)

        v_strain = torch.reshape(torch.matmul(b_glob,v_u), [d_sigma.shape[0], T_PTS, n_elems, 3])
        v_strain = torch.reshape(v_strain, d_sigma.shape)

        return v_u, v_disp, v_strain

    def get_sbvfs(model, dataloader, isTrain = True):

        mdl = copy.deepcopy(model)
        # cProfile.runctx('param_deltas(mdl)',{'mdl':mdl,'param_deltas':param_deltas},{})
        param_dicts = param_deltas(mdl)
        n_vfs = len(param_dicts)

        model.eval()
        with torch.no_grad():

            num_batches = len(dataloader)
            t_pts = dataloader.t_pts
            n_elems = batch_size // t_pts

            iterator = iter(dataloader)
            data = [next(iterator) for i in range(num_batches)]

            x = {list(set(data[i][-2]))[0]: (torch.from_numpy(data[i][0]), torch.from_numpy(data[i][-1][['d_exx','d_eyy','d_exy']].values)) for i in range(num_batches)}

            if isTrain:
                s = {k: model(v[0]).detach() for k,v in x.items()}
            else:
                s = {k: model(torch.from_numpy(dataloader.scaler_x.transform(v[0].numpy()))).detach() for k,v in x.items()}

        s = {k: stress_from_cholesky(v, x[k][1], t_pts, n_elems, sbvf_gen=True) for k,v in s.items()}
        z = list(zip(*x.values()))
        eps = torch.stack(z[0])
        de = torch.stack(z[1])
        sigma = torch.stack(list(s.values()))

        #cProfile.runctx('map(lambda p_dict: sigma_deltas(p_dict,model=mdl,x=eps,d_e=de,s=sigma), param_dicts)',{'sigma_deltas':sigma_deltas,'mdl':mdl,'eps':eps,'de':de,'sigma':sigma,'param_dicts':param_dicts},{})
        ds = torch.stack(list(map(lambda p_dict: sigma_deltas(p_dict,model=mdl,x=eps,d_e=de,s=sigma), param_dicts))).permute(1,0,2,3)

        for i,(k,v) in enumerate(x.items()):
            #Profile.runctx('sbv_fields(ds[i], b_glob, b_inv, n_elems, bcs, active_dof)',{'sbv_fields':sbv_fields,'ds':ds,'i':i,'b_glob':b_glob,'b_inv':b_inv,'n_elems':n_elems,'bcs':bcs,'active_dof':active_dof},{})
            v_u, v_disp, v_strain = sbv_fields(ds[i], b_glob, b_inv, n_elems, bcs, active_dof)
            if isTrain:
                VFs_train[k]['u'] = v_u.numpy()
                VFs_train[k]['v_u'] = v_disp.numpy()
                VFs_train[k]['e'] = torch.reshape(v_strain,[n_vfs,t_pts,n_elems,3]).numpy()
            else:
                VFs_test[k]['u'] = v_u.numpy()
                VFs_test[k]['v_u'] = v_disp.numpy()
                VFs_test[k]['e'] = torch.reshape(v_strain,[n_vfs,t_pts,n_elems,3]).numpy()
    
    def get_yield(e):
        
        window = 3
        der2 = savgol_filter(e, window_length=window, polyorder=2, deriv=2)
        peaks, _ = find_peaks(np.abs(der2),prominence=np.mean(np.abs(der2)))       
        max_der2 = np.max(np.abs(der2[peaks]))
        #max_der2 = np.max(np.abs(der2))
        large = np.where(np.abs(der2) == max_der2)[0]
        gaps = np.diff(large) > window
        begins = np.insert(large[1:][gaps], 0, large[0])
        ends = np.append(large[:-1][gaps], large[-1])
        yield_pt = ((begins+ends)/2).astype(np.int)
        plt.plot(der2)
        plt.plot(peaks,der2[peaks],'og')
        plt.show()
        
        return yield_pt

    def strain_decomp(t, e):

        yield_pt = get_yield(e[:,0])[0]
        for i in range(e.shape[-1]):
            pt = get_yield(e[:,i])
            plt.plot(e[:,i])
            plt.plot(pt, e[:,i][pt], 'ro')
            plt.show()

        e_e = torch.zeros_like(e)
        e_e[:yield_pt,:] = e[:yield_pt,:]
        e_p = e - e_e
        p = torch.cumsum(e_p,axis=0)
        e_dot = torch.diff(e,axis=0)/torch.diff(t,axis=0).repeat(1,3)
        p_dot = torch.diff(p,axis=0)/torch.diff(t,axis=0).repeat(1,3)
        e_p_dot = torch.diff(e_p,axis=0)/torch.diff(t,axis=0).repeat(1,3)

        return e_dot, e_p, p, p_dot, e_p_dot, yield_pt

    def train_loop(dataloader, model, loss_fn, optimizer):
        '''
        Custom loop for neural network training, using mini-batches

        '''

        # Initializing container variables
        num_batches = len(dataloader)
        losses = torch.zeros(num_batches)
        l0_loss = torch.zeros(num_batches)
        l1_loss = torch.zeros(num_batches)
        ch_loss = torch.zeros(num_batches)
        err_stress = torch.zeros(num_batches)
        err_h = torch.zeros(num_batches)
        v_work_real = torch.zeros(num_batches)

        g_norm = torch.zeros(num_batches)

        # Initializing shape variables
        t_pts = dataloader.t_pts
        n_elems = batch_size // t_pts

        model.train()

        for batch in range(num_batches):

            # Extracting variables for training
            X_train, y_train, f_train, coord, tag, inc = dataloader[batch]

            # Converting to pytorch tensors
            X_train = torch.from_numpy(X_train)
            y_train = torch.from_numpy(y_train)
            f_train = torch.from_numpy(f_train) + 0.0
            coord_torch = torch.from_numpy(coord)

            tags = tag[::n_elems].values.tolist()
            incs = inc['inc'][::n_elems].values.tolist()
            t_ = torch.reshape(torch.from_numpy(inc['t'].values),[t_pts,n_elems])
            d_e = torch.from_numpy(inc[['d_exx','d_eyy','d_exy']].values)


            # Extracting element area
            area = torch.reshape(coord_torch[:,4],[batch_size,1])

            # Computing model stress prediction
            pred=model(X_train)

            s, L, H = stress_from_cholesky(pred, d_e, t_pts, n_elems)
        
            #-----------------------------------------------------------------------------------------
            # Loading sensitivity-based virtual fields
            tag = tags[0]
            v_disp = torch.from_numpy(VFs_train[tag]['v_u'])
            v_disp = v_disp[:,incs]
            n_vfs = v_disp.shape[0]
            v_strain = torch.from_numpy(VFs_train[tag]['e'])
            v_strain = v_strain[:,incs]
            v_strain = torch.reshape(v_strain,(n_vfs,s.shape[0],s.shape[1]))

            #----------------------------------------------------------------------------------------

            # Computing predicted virtual work
            int_work = torch.sum(torch.sum(torch.reshape((s * v_strain * area * ELEM_THICK),[n_vfs,t_pts,n_elems,3]),-1),-1,keepdim=True)

            # Computing real virtual work
            int_work_real = torch.sum(torch.sum(torch.reshape((y_train * v_strain * area * ELEM_THICK),[n_vfs,t_pts,n_elems,3]),-1),-1,keepdim=True)

            f = f_train[:,:2][::n_elems,:]

            # Computing external virtual work
            ext_work = torch.sum(torch.reshape(f,[t_pts,1,2])*v_disp,-1)

            # # Specify L1 and L2 weights
            # l1_weight = 0.5
            # l2_weight = 0.5
            # parameters = []
            # for parameter in model.named_parameters():
            #     if 'weight' in parameter[0]:
            #         parameters.append(parameter[1].view(-1))
            # # l1 = l1_weight * torch.abs(torch.cat(parameters)).sum()
            # l2 = l2_weight * torch.square(torch.cat(parameters)).sum()

            #Initial loss
            idx_0 = np.where(np.array(incs)==0)[0][0]*n_elems
            s_0 = s[idx_0:idx_0+n_elems,:] 
            
            l_0 = torch.mean(torch.square(s_0))

            # Intrinsic dissipation
            l_1 = torch.mean(torch.square(torch.nn.functional.relu(-(s * d_e/0.02))))

            # Positive definiteness of H - positive diagonal of L
            l_cholesky = torch.mean(torch.square(nn.functional.relu(-torch.diagonal(L, offset=0, dim1=2, dim2=3))))
            
            # #Equilibrium
            # i_f = torch.sum(torch.reshape(pred[:,:2] * area * ELEM_THICK / LENGTH,[T_PTS,n_elems,2]),1)
            # l_f = mse(i_f,f)
            # e = dataloader.scaler_x.inverse_transform(X_train)
            # e = torch.reshape(torch.from_numpy(e[:,::2]),[t_pts,n_elems,3])
            # de = torch.reshape(torch.diff(e,dim=0),[-1,3])
            # ds = torch.reshape(torch.diff(torch.reshape(pred,[t_pts,n_elems,3]),dim=0),[-1,3])

            # l_1 = torch.mean(torch.nn.functional.relu(-(ds[:,0]*de[:,0]))) + torch.mean(torch.nn.functional.relu(-(ds[:,1]*de[:,1]))) + torch.mean(torch.nn.functional.relu(-(ds[:,1]*de[:,1])))
            # #Plastic power
            
            # wp = torch.sum(torch.reshape(pred * ep_dot,[t_pts,n_elems,3]) ,-1)
            # l_wp = torch.sum(torch.nn.functional.relu(-wp))
            # f_i = torch.sum(torch.reshape((pred * area * ELEM_THICK/LENGTH),[t_pts,n_elems,3]),1)
            # f_g = f_train[:,:][::n_elems,:]

            # h = batch_jacobian(model,X_train)
            # h = torch.flatten(h,1,2)
            # l_h = torch.sum(torch.square(torch.nn.functional.relu(-h)))

            # # Computing loss
            loss = loss_fn(int_work,ext_work) + 100*l_cholesky
            #loss = mse(f_i,f_g) + l_00.001*l_1
            cost = loss_fn(int_work_real, ext_work)
            
            # Backpropagation and weight's update
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # # Gradient clipping - as in https://github.com/pseeth/autoclip
            g_norm[batch] = get_grad_norm(model)
            #clip_value = np.percentile(g_norm, 10)
            #nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            #nn.utils.clip_grad_value_(model.parameters(),10.0)

            optimizer.step()
            #scheduler.step()
            # for i, layer in enumerate(model.layers):
            #     if i != 0:
            #         layer.apply(constraints)
            

            if tags[0]==train_trials[0]:
                numSteps = math.floor(SCALE_PAR * int_work.shape[1])
                if numSteps == 0:
                    numSteps = 1
                ivw_sort = torch.sort(torch.abs(int_work.detach()),1,descending=True).values
            
                alpha = (1/torch.mean(ivw_sort[:,0:numSteps],1))

            # Saving loss values
            losses[batch] = loss.detach().item()
            l0_loss[batch] = l_0.detach().item()
            l1_loss[batch] = l_1.detach().item()
            ch_loss[batch] = l_cholesky.detach().item()

            err_stress[batch] = mse(s.detach(),y_train)
            
            v_work_real[batch] = cost.detach().item()
            #v_work_real[batch] = 0
            #g_norm.append(get_grad_norm(model))
            #l += loss
            # for i,(n,j) in enumerate(tuple(zip(tags,incs))):
            #     #VFs[n]['u'][j] = v_u[:,i].detach().numpy()
            #     #VFs[n]['e'][j] = torch.reshape(v_strain,[n_vfs,t_pts,n_elems,3])[:,i].detach().numpy()
            #     w_virt[n]['w_int'][j] = torch.sum(int_work[:,i].detach()).numpy()
            #     w_virt[n]['w_int_real'][j] = torch.sum(int_work_real[:,i].detach()).numpy()
            #     w_virt[n]['w_ext'][j] = torch.sum(ext_work[:,i].detach()).numpy()

            print('\r>Train: %d/%d' % (batch + 1, num_batches), end='')
        #l.backward()
        #optimizer.step()
        #scheduler.step()
        
        
        # get_sbvfs(copy.deepcopy(model), sbvf_generator)
        # return losses, v_work_real, v_disp, v_strain, v_u
        #-----------------------------
        return losses, l0_loss, err_stress, v_work_real, g_norm, alpha, l1_loss, ch_loss

    def test_loop(dataloader, model, loss_fn):

        #global w_virt

        num_batches = len(dataloader)
        test_losses = torch.zeros(num_batches)

        t_pts = dataloader.t_pts
        n_elems = batch_size//t_pts
        #n_vfs = v_strain.shape[0]

        model.eval()
        
        with torch.no_grad():

            for batch in range(num_batches):

                # Extracting variables for testing
                X_test, y_test, f_test, coord, tag, inc = dataloader[batch]

                # Converting to pytorch tensors
                if dataloader.scaler_x != None:
                    X_test = torch.from_numpy(dataloader.scaler_x.transform(X_test))
                else:
                    X_test = torch.tensor(X_test, dtype=torch.float64)

                y_test = torch.from_numpy(y_test)
                f_test = torch.from_numpy(f_test) + 0.0
                coord_torch = torch.from_numpy(coord)

                tags = tag[::n_elems].values.tolist()
                incs = inc['inc'][::n_elems].values.tolist()

                area = torch.reshape(coord_torch[:,4],[batch_size,1])

                d_e = torch.from_numpy(inc[['d_exx','d_eyy','d_exy']].values)

                pred = model(X_test)
                s, _, _ = stress_from_cholesky(pred, d_e, t_pts, n_elems)
                #--------------------------------------
                # s_ = model_2(X_test).detach()
                # f_ = torch.sum(torch.reshape(s_ * area * ELEM_THICK / LENGTH,[t_pts,n_elems,3]),1)[:,:2]
                #---------------------------------------------------------------

                #-----------------------------------------------------------------------------------------
                # Loading sensitivity-based virtual fields
                tag = tags[0]
                v_disp = torch.from_numpy(VFs_test[tag]['v_u'])
                n_vfs = v_disp.shape[0]
                v_strain = torch.from_numpy(VFs_test[tag]['e'])
                v_strain = torch.reshape(v_strain,(n_vfs,s.shape[0],s.shape[1]))
                #-----------------------------------------------------------------------------------------

                int_work = torch.sum(torch.sum(torch.reshape((s * v_strain * area * ELEM_THICK),[n_vfs,t_pts,n_elems,3]),-1),-1,keepdim=True)

                int_work_real = torch.sum(torch.sum(torch.reshape((y_test * v_strain * area * ELEM_THICK),[n_vfs,t_pts,n_elems,3]),-1),-1,keepdim=True)

                f = f_test[:,:2][::n_elems,:]

                ext_work = torch.sum(torch.reshape(f,[t_pts,1,2])*v_disp,-1)


                #f_i = torch.sum(torch.reshape((pred * area * ELEM_THICK/LENGTH),[t_pts,n_elems,3]),1)
                #f_g = f_test[:,:][::n_elems,:]
                # Computing losses
                test_loss = loss_fn(int_work, ext_work)
                #test_loss=mse(f_i,f_g)
                
                test_losses[batch] = test_loss

                # for i,(n,j) in enumerate(tuple(zip(tags,incs))):
                #     #VFs[n]['u'][j] = v_u[:,i].detach().numpy()
                #     #VFs[n]['e'][j] = torch.reshape(v_strain,[n_vfs,t_pts,n_elems,3])[:,i].detach().numpy()
                #     w_virt[n]['w_int'][j] = torch.sum(int_work[:,i].detach()).numpy()
                #     w_virt[n]['w_int_real'][j] = torch.sum(int_work_real[:,i].detach()).numpy()
                #     w_virt[n]['w_ext'][j] = torch.sum(ext_work[:,i].detach()).numpy()

                print('\r>Test: %d/%d' % (batch + 1, num_batches), end='')

        #get_sbvfs(copy.deepcopy(model), dataloader, isTrain=False)
        return test_losses
#----------------------------------------------------
    
    # Default floating point precision for pytorch
    torch.set_default_dtype(torch.float64)

    # Specifying random seed
    os.environ['PYTHONHASHSEED']=str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Reading mesh file
    mesh, connectivity, dof = read_mesh(TRAIN_MULTI_DIR)

    # Defining geometry limits
    x_min = min(mesh[:,1])
    x_max = max(mesh[:,1])
    y_min = min(mesh[:,-1])
    y_max = max(mesh[:,-1])

    # Total degrees of freedom
    total_dof = mesh.shape[0] * 2

    # Defining edge boundary conditions
    #     0 - no constraint
    #     1 - displacements fixed along the edge
    #     2 - displacements constant along the edge
    bcs = {
        'left': {
            'cond': [1,0],
            'dof': global_dof(mesh[mesh[:,1]==x_min][:,0])},
        'bottom': {
            'cond': [0,1],
            'dof': global_dof(mesh[mesh[:,-1]==y_min][:,0])},
        'right': {
            'cond': [2,0],
            'dof': global_dof(mesh[mesh[:,1]==x_max][:,0])},
        'top': {
            'cond': [0,0],
            'dof': global_dof(mesh[mesh[:,-1]==y_max][:,0])}
    }

    # Constructing element properties based on mesh info
    elements = [Element(connectivity[i,:],mesh[connectivity[i,1:]-1,1:],dof[i,:]) for i in range(connectivity.shape[0])]

    # Assembling global strain-displacement matrices
    b_glob, b_inv, active_dof = global_strain_disp(elements, total_dof, bcs)

    # Loading data
    df_list, _ = load_dataframes(TRAIN_MULTI_DIR)

    # Merging training data
    data = pd.concat(df_list, axis=0, ignore_index=True).dropna(axis=0)

    T_PTS = len(set(data['t']))

    # Performing test/train split
    partition = {"train": None, "test": None}

    # if T_PTS==DATA_SAMPLES:
    # Reorganizing dataset by tag, subsequent grouping by time increment
    data_by_tag = [df for _, df in data.groupby(['tag'])]
    random.shuffle(data_by_tag)
    data_by_t = [[df for _, df in group.groupby(['t'])] for group in data_by_tag]
    #random.shuffle(data_by_t)
    data_by_batches = list(itertools.chain(*data_by_t))
    #random.shuffle(data_by_batches)

    batch_size = len(data_by_batches[0]) * T_PTS

    data = pd.concat(data_by_batches).reset_index(drop=True)

    trials = list(set(data['tag'].values))
    random.shuffle(trials)
    test_trials = random.sample(trials, math.floor(len(trials)*TEST_SIZE))
    train_trials = list(set(trials).difference(test_trials))

    partition['train'] = data[data['tag'].isin(train_trials)].index.tolist()
    partition['test'] = data[data['tag'].isin(test_trials)].index.tolist()

    # Selecting model features
    X, y, f, coord, info = select_features_multi(data)
    
    # Preparing data generators for mini-batch training
    #scaler_x = joblib.load('outputs/9-elem-50-plastic_sbvf_abs_direct/models/[6-20x3-3]-9-elem-50-plastic-1163-VFs-scaler_x_all_const.pkl')
    train_generator = DataGenerator(X, y, f, coord, info, partition["train"], batch_size, shuffle=False, std=True, t_pts=T_PTS)
    sbvf_generator = DataGenerator(X, y, f, coord, info, partition["train"], batch_size, shuffle=False, std=True, t_pts=T_PTS)

    test_generator = DataGenerator(X, y, f, coord, info, partition['test'], batch_size, shuffle=False, std=False, t_pts=T_PTS, scaler=train_generator.scaler_x)
   

    # Model variables
    N_INPUTS = X.shape[1]
    N_OUTPUTS = y.shape[1] + 3
   
    N_UNITS = [12,12,12]
    H_LAYERS = len(N_UNITS)

    INCREMENTAL_VFS = False

    model_1 = NeuralNetwork(N_INPUTS, N_OUTPUTS, N_UNITS, H_LAYERS, b_norm=False)
    #model_1 = InputConvexNN(N_INPUTS, N_OUTPUTS, N_UNITS, H_LAYERS)

    #model_2 = NeuralNetwork(6, 3, [20,20,20], 3)
    #model_2.load_state_dict(torch.load('outputs/9-elem-50-plastic_sbvf_abs_direct/models/[6-20x3-3]-9-elem-50-plastic-1163-VFs_all_const.pt'))
    #model_1.load_state_dict(torch.load('outputs/9-elem-50-plastic_sbvf_abs_direct/models/[6-20x3-3]-9-elem-50-plastic-1163-VFs_all_const.pt'))
    model_1.apply(init_weights)
    # for p in model_1.parameters():
    #     p.register_hook(lambda grad: torch.clamp(grad, -1.0, 1.0))
    # geotorch.symmetric(model_1.layers[-1], "weight")

    # clip_value = 250
    # for p in model_1.parameters():
    #     p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))
    

    # Training variables
    epochs=10000

    # Optimization variables
    L_RATE = 0.08
    LR_MULT = 1.0

    params = layer_wise_lr(model_1, lr_mult=LR_MULT, learning_rate=L_RATE)

    SCALE_PAR = 0.1
    loss_fn = SBVFLoss(scale_par=SCALE_PAR,res_scale=False)
    
    mse = torch.nn.MSELoss()

    WEIGH_DECAY = 0.001
    optimizer = torch.optim.AdamW(params=params, weight_decay=WEIGH_DECAY)

    

    # lr_lambda = lambda x: math.exp(x * math.log(1e-7 / 1.0) / (epochs * len(train_generator)))
    #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    PATIENCE = 65
    COOLDOWN = 20
    FACTOR = 0.8
    MIN_LR = 0.0001
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=PATIENCE, cooldown=COOLDOWN, factor=FACTOR, min_lr=[params[i]['lr']*MIN_LR for i in range(len(params))])

    #scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,base_lr=1e-3, max_lr=0.08, mode='exp_range',gamma=0.99994,cycle_momentum=False,step_size_up=len(train_generator)*50,step_size_down=len(train_generator)*50)
    #scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,base_lr=1e-3,max_lr=0.008,mode='exp_range',gamma=0.99994,cycle_momentum=False)
    
    constraints=weightConstraint(cond='plastic')

    # Container variables for history purposes
    train_loss = []
    
    err_stresses = []
    v_work = []
    val_loss = []
    epochs_ = []
    
    l0_loss = []
    l1_loss = []
    ch_loss = []

    g_norms = []
    # Initializing the early_stopping object
    #early_stopping = EarlyStopping(patience=1000, path='temp/checkpoint.pt', verbose=True)

    VFs_train = {key: {k: None for k in ['u','e','v_u']} for key in train_trials}
    VFs_test = {key: {k: None for k in ['u','e','v_u']} for key in test_trials}

    w_virt = {key: {k: dict.fromkeys(set(info['inc'])) for k in ['w_int','w_int_real','w_ext']} for key in set(info['tag'])}

    k_opt = 0
    l_opt = 0
  

    ################################################################################
    #                           WEIGHTS & BIASES LOGGING
    ################################################################################

    WANDB_LOG = True

    PROJECT = 'indirect_training_cholesky'

    if WANDB_LOG:
        config = {
            "inputs": N_INPUTS,
            "outputs": N_OUTPUTS,
            "hidden_layers": H_LAYERS,
            "hidden_units": "/".join(str(x) for x in N_UNITS),
            "incremental_vfs": INCREMENTAL_VFS,
            "epochs": epochs,
            "lr": L_RATE,
            "l2_reg": WEIGH_DECAY,
            "scale_par": SCALE_PAR,
            "sch_patience": PATIENCE,
            "sch_cooldown": COOLDOWN,
            "sch_factor": FACTOR,
            "sch_min_lr": MIN_LR
        }
        
        run = wandb.init(project=f'{PROJECT}', entity="rmbl",config=config)
        run.watch(model_1,log='all')

    ################################################################################

    for t in range(epochs):

        print('\r--------------------\nEpoch [%d/%d]' % (t + 1, epochs))

        epochs_.append(t+1)

        #Shuffling batches
        for generator in [train_generator,test_generator]:
            generator.on_epoch_end()
        
        #--------------------------------------------------------------
        #Calculating VFs
        if t==0:
            print('Computing virtual fields...')
            get_sbvfs(copy.deepcopy(model_1), sbvf_generator)
            get_sbvfs(copy.deepcopy(model_1), test_generator, isTrain=False)
            
        #--------------------------------------------------------------

        # Train loop
        start_train = time.time()

        #--------------------------------------------------------------
        batch_losses, l_0, err_stress, batch_v_work, grad_norm, alpha, l_1, l_ch = train_loop(train_generator, model_1, loss_fn, optimizer)
        #--------------------------------------------------------------

        q.put((w_virt[train_trials[0]],train_trials[0]))

        train_loss.append(torch.mean(batch_losses))
        l0_loss.append(torch.mean(l_0))
        err_stresses.append(torch.mean(err_stress))
        v_work.append(torch.mean(batch_v_work))
        l1_loss.append(torch.mean(l_1))
        ch_loss.append(torch.mean(l_ch))
        g_norms.append(torch.mean(grad_norm))

        end_train = time.time()

        #Apply learning rate scheduling if defined
        try:
            pass
            scheduler.step(train_loss[t])
            print('. t_loss: %.6e -> lr: %.3e // [s_mse] - > %.3e | [v_work] -> %.3e -- %.3fs' % (train_loss[t], scheduler._last_lr[0], err_stresses[t], v_work[t], end_train - start_train))
        except:
            print('. t_loss: %.6e // [v_work] -> %.3e -- %.3fs' % (train_loss[t], v_work[t], end_train - start_train))

        # Test loop
        start_test = time.time()

        #-----------------------------------------------------------------------------------
        batch_val_losses = test_loop(test_generator, model_1, loss_fn)
        #-----------------------------------------------------------------------------------

        val_loss.append(torch.mean(batch_val_losses).item())

        end_test = time.time()

        print('. v_loss: %.6e -- %.3fs' % (val_loss[t], end_test - start_test))
        q.put(([train_loss[t],val_loss[t],l0_loss[t],err_stresses[t]],'loss'))

        # if t > 200:
        #     if early_stopping.counter==0:
        #         w_virt_best = copy.deepcopy(w_virt)
        #         VFs_train_best = copy.deepcopy(VFs_train)
        #     # Check validation loss for early stopping
        #     early_stopping(val_loss[t], model_1)

        if t!= 0:

            #tol_updt = (train_loss[0]/2.3**(k_opt))
            tol_updt = (1/2.3**k_opt)
            delta_loss = train_loss[t]-train_loss[t-1]
            
            if delta_loss > 0:
                l_opt +=1
            else:
                l_opt = 0

            print('\n[k_opt: %i | l_opt: %i] -- grad_norm (mean): %.6e | delta_loss: %.6e | tol_updt: %.6e' % (k_opt, l_opt, g_norms[t], delta_loss, tol_updt))
            
            #if t % 4999 == 0:
            if g_norms[t] <= tol_updt:
            #if (abs(delta_loss) < tol_updt):
            #if (t%99==0):

                print('\nUpdating virtual fields [%i]' % (k_opt))

                get_sbvfs(copy.deepcopy(model_1), sbvf_generator)

                get_sbvfs(copy.deepcopy(model_1), test_generator, isTrain=False)
            
                k_opt += 1
                l_opt = 0

        # if (t!= 0 and delta_loss < 1e-9) or early_stopping.early_stop:
        #     print("Early stopping")
        #     break
        vfs = alpha.shape[0]
        
        if WANDB_LOG:
            wandb.log({
                'epoch': t,
                'l_rate': scheduler._last_lr[0],
                'grad_norm': g_norms[t],
                'train_loss': train_loss[t],
                'test_loss': val_loss[t],
                'mse_stress': err_stresses[t],
                's(0)_error': l0_loss[t],
                'dissipation_error': l1_loss[t],
                'cholesky_error': ch_loss[t],
                'vf_update': k_opt,
                'alpha_0': alpha[0],
                'alpha_%i' % (math.floor(vfs/4)): alpha[math.floor(vfs/4)],
                'alpha_%i' % (math.floor(vfs/2)): alpha[math.floor(vfs/2)],
                'alpha_%i' % (math.floor(vfs*3/4)): alpha[math.floor(vfs*3/4)],
                'alpha_%i' % (vfs): alpha[-1]
            })
        

    q.put('Q')
    print("Done!")

    # load the last checkpoint with the best model
    #model_1.load_state_dict(torch.load('temp/checkpoint.pt'))

    epochs_ = np.reshape(np.array(epochs_), (len(epochs_),1))
    train_loss = np.reshape(np.array(train_loss), (len(train_loss),1))
    val_loss = np.reshape(np.array(val_loss), (len(val_loss),1))

    try:
        history = pd.DataFrame(np.concatenate([epochs_, train_loss, val_loss], axis=1), columns=['epoch','loss','val_loss'])
    except:
        pass


    task = r'%s-[%i-%ix%i-%i]-%s-%i-VFs' % (run.name,N_INPUTS, N_UNITS[0], H_LAYERS, N_OUTPUTS, TRAIN_MULTI_DIR.split('/')[-2], count_parameters(model_1))

    output_task = 'outputs/' + TRAIN_MULTI_DIR.split('/')[-2] + '_sbvf_abs_cholesky'
    output_loss = output_task + '/loss/'
    output_stats = output_task + '/stats/'
    output_models = output_task + '/models/'
    output_val = output_task + '/val/'
    output_logs = output_task + '/logs/'

    directories = [output_task, output_loss, output_stats, output_models, output_val, output_logs]

    for dir in directories:
        try:
            os.makedirs(dir)

        except FileExistsError:
            pass

    history.to_csv(output_loss + task + '.csv', sep=',', encoding='utf-8', header='true')

    plot_history(history, output_loss, True, task)

    torch.save(model_1.state_dict(), output_models + task + '.pt')
    np.save(output_logs + 'sbvfs.npy', VFs_train)
    np.save(output_logs + 'w_virt.npy', w_virt)
    #joblib.dump([VFs, W_virt], 'sbvfs.pkl')
    if train_generator.std == True:
        joblib.dump(train_generator.scaler_x, output_models + task + '-scaler_x.pkl')
        time.sleep(1)

    if WANDB_LOG:
        # 3️⃣ At the end of training, save the model artifact
        # Name this artifact after the current run
        task_ = r'__%i-%ix%i-%i__%s-%i-VFs' % (N_INPUTS, N_UNITS[0], H_LAYERS, N_OUTPUTS, TRAIN_MULTI_DIR.split('/')[-2], count_parameters(model_1))
        model_artifact_name = run.id + '_' + run.name + task_
        # Create a new artifact, which is a sample dataset
        model = wandb.Artifact(model_artifact_name, type='model')
        # Add files to the artifact, in this case a simple text file
        model.add_file(local_path=output_models + task + '.pt')
        model.add_file(output_models + task + '-scaler_x.pkl')
        # Log the model to W&B
        run.log_artifact(model)
        # Call finish if you're in a notebook, to mark the run as done
        run.finish()

# -------------------------------
#           Main script
# -------------------------------
if __name__ == '__main__':
    # Creating temporary folder
    try:
        os.makedirs('./temp')
    except FileExistsError:
        pass

    # # Default floating point precision for pytorch
    # torch.set_default_dtype(torch.float64)

    # # Specifying random seed
    # random.seed(SEED)
    # torch.manual_seed(SEED)
    window=Tk()
    #Create a queue to share data between process
    q = multiprocessing.Queue()

    #Create and start the simulation process
    train_ = multiprocessing.Process(None,train,args=(q,))
    train_.start()

    #Create the base plot
    plot()

    #Call a function to update the plot when there is new data
    updateplot(q)

    window.mainloop()

    # Deleting temp folder
    try:
        shutil.rmtree('temp/')
    except FileNotFoundError:
        pass

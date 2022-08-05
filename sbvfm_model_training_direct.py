# ---------------------------------
#    Library and function imports
# ---------------------------------
from audioop import mul
import cProfile, pstats
import os
import shutil
import joblib
from sklearn import model_selection
from constants import *
from functions import (
    InputConvexNN,
    sbvf_loss,
    load_dataframes,
    prescribe_u,
    select_features_multi,
    standardize_data,
    plot_history,
    read_mesh,
    global_strain_disp,
    param_deltas,
    global_dof)
from functions import (
    weightConstraint,
    EarlyStopping,
    NeuralNetwork,
    Element)
from functools import partial
import copy
from torch import nn
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
        self.t = info['inc'].iloc[list_IDs].reset_index(drop=True)
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

    global w_ext,w_int,w_int_real,t_loss,v_loss,s_loss,s_err,ax,ax_2,canvas

    fig = matplotlib.figure.Figure(figsize=(15, 5))

    ax = fig.add_subplot(1,2,1)
    ax_2 = fig.add_subplot(1,2,2)
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()
    canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)
    canvas._tkcanvas.pack(side=TOP, fill=BOTH, expand=1)

    w_ext, = ax.plot([],label='w_ext')
    w_int, = ax.plot([],label='w_int_ann')
    w_int_real, = ax.plot([],label='w_int')
    t_loss, = ax_2.plot([],label='train_loss')
    v_loss, = ax_2.plot([],label='test_loss')
    s_loss, = ax_2.plot([],label='s(0)_loss')
    s_err, = ax_2.plot([],label='mse_stress')
    ax_2.set_yscale('log')
    ax_2.set_xlim([0, 2])
    ax_2.set_title('Loss curves')
    ax_2.set_xlabel('Epochs')
    ax_2.set_ylabel('Loss')
    ax_2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(loc='best')
    ax_2.legend(loc='best')



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

                lower = min([min(a) for a in [d1,d2,d3,d4]])
                upper = max([max(a) for a in [d1,d2,d3,d4]])

                ax_2.set_xlim([0, max(epochs)+5])
                ax_2.set_ylim([lower-0.5*abs(lower), upper+0.1*abs(upper)])

                ax_2.draw_artist(ax_2.patch)
                ax_2.draw_artist(t_loss)
                ax_2.draw_artist(v_loss)
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
    return jacobian(f_sum, x,create_graph=True).permute(1,0,2)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

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
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        return total_norm

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
            #torch.nn.init.ones_(m.bias)
            m.bias.data.fill_(0.01)

    # Defined at top-level because nested functions are not importable in multiprcessing.pool
    # Refer to: https://stackoverflow.com/questions/52265120/python-multiprocessing-pool-attributeerror
    def sigma_deltas(param_dict, model, x, s):

        s_=s.reshape((-1,)+s.shape[2:])
        x_=x.reshape((-1,)+x.shape[2:])
        with torch.no_grad():
            model.load_state_dict(param_dict)
            d_sigma = (s_ - model(x_).detach())

        d_sigma = torch.reshape(d_sigma,s.shape)

        if INCREMENTAL_VFS:
            ##### INCREMENTAL DELTA STRESS #############
            n_elems = s.shape[1] // T_PTS
            ds_ = torch.reshape(d_sigma,[d_sigma.shape[0],T_PTS,n_elems,3])
            
            dd_sigma = torch.zeros_like(ds_)
            dd_sigma[:,1:] = (ds_[:,1:] - ds_[:,:-1]) / 0.02

            return torch.reshape(dd_sigma,d_sigma.shape)
            # # ##############################################
        else:
            return d_sigma

    # def sigma_deltas(model, X_train, pred, param_dicts, incs, idx=None):

    #     delta_stress = torch.zeros((len(param_dicts), pred.shape[0], pred.shape[1]))
    #     mdl = copy.deepcopy(model)
    #     mdl.eval()
    #     with torch.no_grad():
    #         for i,param_dict in enumerate(param_dicts):
    #             mdl.load_state_dict(param_dict)
    #             pred_eval = mdl(X_train)
    #             delta_stress[i,:,:] = (pred - pred_eval)

    #     ##### INCREMENTAL DELTA STRESS #############
    #     # n_vfs = delta_stress.shape[0]
    #     # n_elems = delta_stress.shape[1] // T_PTS
    #     # d_stress = torch.reshape(delta_stress,[n_vfs,T_PTS,n_elems,3])
    #     # if idx != None:
    #     #     d_stress = d_stress[:,idx]
    #     # d_s = torch.zeros_like(d_stress)
    #     # d_s[:,1:] = (d_stress[:,1:] - d_stress[:,:-1]) / 0.02

    #     # return torch.reshape(d_s[:,incs],delta_stress.shape)
    #     # # ##############################################
    #     return delta_stress

    # def sigma_deltas(model, x, s, param_dict):

    #     delta_stress = torch.zeros_like(s)
    #     model.eval()
    #     with torch.no_grad():

    #         model.load_state_dict(param_dict)
    #         ds = model(x)
    #         d_sigma = (s - ds)

    #     ##### INCREMENTAL DELTA STRESS #############
    #     # n_vfs = delta_stress.shape[0]
    #     # n_elems = delta_stress.shape[1] // T_PTS
    #     # d_stress = torch.reshape(delta_stress,[n_vfs,T_PTS,n_elems,3])
    #     # if idx != None:
    #     #     d_stress = d_stress[:,idx]
    #     # d_s = torch.zeros_like(d_stress)
    #     # d_s[:,1:] = (d_stress[:,1:] - d_stress[:,:-1]) / 0.02

    #     # return torch.reshape(d_s[:,incs],delta_stress.shape)
    #     # # ##############################################
    #     return d_sigma

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

            x = {list(set(data[i][-2]))[0]: torch.from_numpy(data[i][0]) for i in range(num_batches)}

            if isTrain:
                s = {k: model(v).detach() for k,v in x.items()}
            else:
                s = {k: model(torch.from_numpy(dataloader.scaler_x.transform(v.numpy()))).detach() for k,v in x.items()}

        eps = torch.stack(list(x.values()))
        sigma = torch.stack(list(s.values()))

        ds = torch.stack(list(map(lambda p_dict: sigma_deltas(p_dict,model=mdl,x=eps,s=sigma), param_dicts))).permute(1,0,2,3)

        for i,(k,v) in enumerate(x.items()):
            v_u, v_disp, v_strain = sbv_fields(ds[i], b_glob, b_inv, n_elems, bcs, active_dof)
            if isTrain:
                VFs_train[k]['u'] = v_u.numpy()
                VFs_train[k]['v_u'] = v_disp.numpy()
                VFs_train[k]['e'] = torch.reshape(v_strain,[n_vfs,t_pts,n_elems,3]).numpy()
            else:
                VFs_test[k]['u'] = v_u.numpy()
                VFs_test[k]['v_u'] = v_disp.numpy()
                VFs_test[k]['e'] = torch.reshape(v_strain,[n_vfs,t_pts,n_elems,3]).numpy()

    def get_sbvfs_batch(model, x, tag):

        mdl = copy.deepcopy(model)
        # cProfile.runctx('param_deltas(mdl)',{'mdl':mdl,'param_deltas':param_deltas},{})
        param_dicts = param_deltas(mdl)

        n_vfs = len(param_dicts)

        n_elems = batch_size // T_PTS

        model.eval()
        with torch.no_grad():

            s = torch.stack([model(x_).detach() for i,x_ in enumerate(list(x))])

        ds = torch.stack(list(map(lambda p_dict: sigma_deltas(p_dict,model=mdl,x=torch.stack(list(x)),s=s), param_dicts))).permute(1,0,2,3)

        for i in range(len(tag)):
            v_u, v_disp, v_strain = sbv_fields(ds[i], b_glob, b_inv, n_elems, bcs, active_dof)

            VFs_train[tag[i]]['u'] = v_u.numpy()
            VFs_train[tag[i]]['v_u'] = v_disp.numpy()
            VFs_train[tag[i]]['e'] = torch.reshape(v_strain,[n_vfs,T_PTS,n_elems,3]).numpy()


    def train_loop(dataloader, model, loss_fn, optimizer):
        '''
        Custom loop for neural network training, using mini-batches

        '''

        num_batches = len(dataloader)
        losses = torch.zeros(num_batches)
        l_stress = torch.zeros(num_batches)
        err_stress = torch.zeros(num_batches)
        v_work_real = torch.zeros(num_batches)

        g_norm = []

        t_pts = dataloader.t_pts
        n_elems = batch_size // t_pts

        l = 0
        model.train()
        #optimizer.zero_grad(set_to_none=True)
        for batch in range(num_batches):

            # Extracting variables for training
            X_train, y_train, f_train, coord, tag, inc = dataloader[batch]

            # Converting to pytorch tensors
            X_train = torch.reshape(torch.from_numpy(X_train),[t_pts,-1])
            y_train = torch.from_numpy(y_train)
            f_train = torch.from_numpy(f_train) + 0.0
            coord_torch = torch.from_numpy(coord)

            tags = tag[::n_elems].values.tolist()
            incs = inc[::n_elems].values.tolist()

            # Extracting element area
            area = torch.reshape(coord_torch[:,4],[batch_size,1])

            # Computing model stress prediction
            pred=model(X_train)

            #-----------------------------------------------------------------------------------------
            # Loading sensitivity-based virtual fields
            tag = tags[0]
            v_disp = torch.from_numpy(VFs_train[tag]['v_u'])[sbvf_idx]
            v_disp = v_disp[:,incs]
            n_vfs = v_disp.shape[0]
            v_strain = torch.from_numpy(VFs_train[tag]['e'])[sbvf_idx]
            v_strain = v_strain[:,incs]
            v_strain = torch.reshape(v_strain,(n_vfs,pred.shape[0],pred.shape[1]))

            #-----------------------------------------------------------------------------------------

            # Computing sensitivity-based virtual fields
            #v_u, v_disp, v_strain = sbv_fields(d_sigma, b_glob, b_inv, n_elems, bcs, active_dof)

            # Computing predicted virtual work
            int_work = torch.sum(torch.reshape((pred * v_strain * area * ELEM_THICK),[n_vfs,t_pts,n_elems,3]),-1)

            # Computing real virtual work
            int_work_real = torch.sum(torch.reshape((y_train * v_strain * area * ELEM_THICK),[n_vfs,t_pts,n_elems,3]),-1)

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

            # l_jac_weight = 0.5
            #jac = batch_jacobian(model,X_train)[:,:,-3:][0]
            #l_jac = torch.sum(torch.nn.functional.relu(-jac))

            # Applying initial condition for stress (S(t0)=0)
            t_steps_idx = torch.sort(torch.tensor(incs)).indices
            s = torch.reshape(pred,[T_PTS,n_elems,3])[t_steps_idx,:]
            l_s = torch.mean(torch.sum(torch.square(s[0]),0))

            # # Monotonic stress
            # dw = torch.zeros_like(int_work)
            # dw[:,1:] = torch.abs(dw[:,:-1]) - torch.abs(dw[:,1:])
            # lw_2 = lambda_*torch.mean(torch.nn.functional.relu(dw))

            # # Computing loss
            loss = loss_fn(int_work,ext_work)
            cost = loss_fn(int_work_real, ext_work)

            # Backpropagation and weight's update
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # # Gradient clipping - as in https://github.com/pseeth/autoclip
            # g_norm.append(get_grad_norm(model))
            # clip_value = np.percentile(g_norm, 25)
            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_value)

            optimizer.step()
            #scheduler.step()
            #model.apply(constraints)

            # Saving loss values
            losses[batch] = loss.detach().item()
            l_stress[batch] = l_s.detach().item()
            err_stress[batch] = torch.mean(torch.sum(torch.square(pred.detach()-y_train),1))
            v_work_real[batch] = cost.detach().item()
            g_norm.append(get_grad_norm(model))
            #l += loss
            for i,(n,j) in enumerate(tuple(zip(tags,incs))):
                #VFs[n]['u'][j] = v_u[:,i].detach().numpy()
                #VFs[n]['e'][j] = torch.reshape(v_strain,[n_vfs,t_pts,n_elems,3])[:,i].detach().numpy()
                w_virt[n]['w_int'][j] = torch.sum(torch.sum(int_work[:,i].detach(),-1)).numpy()
                w_virt[n]['w_int_real'][j] = torch.sum(torch.sum(int_work_real[:,i].detach()),-1).numpy()
                w_virt[n]['w_ext'][j] = torch.sum(torch.sum(ext_work[:,i].detach(),-1)).numpy()

            print('\r>Train: %d/%d' % (batch + 1, num_batches), end='')
        #l.backward()
        #optimizer.step()
        #scheduler.step()
        
        

        # get_sbvfs(copy.deepcopy(model), sbvf_generator)
        # return losses, v_work_real, v_disp, v_strain, v_u
        #-----------------------------
        return losses, l_stress, err_stress, v_work_real, np.mean(g_norm)

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
                if dataloader.scaler_x is not None:
                    X_test = torch.from_numpy(dataloader.scaler_x.transform(X_test))
                else:
                    X_test = torch.tensor(X_test, dtype=torch.float64)

                y_test = torch.from_numpy(y_test)
                f_test = torch.from_numpy(f_test) + 0.0
                coord_torch = torch.from_numpy(coord)

                tags = tag[::n_elems].values.tolist()
                incs = inc[::n_elems].values.tolist()

                area = torch.reshape(coord_torch[:,4],[batch_size,1])

                pred = model(X_test)

                #-----------------------------------------------------------------------------------------
                # Loading sensitivity-based virtual fields
                tag = tags[0]
                v_disp = torch.from_numpy(VFs_test[tag]['v_u'])[sbvf_idx]
                n_vfs = v_disp.shape[0]
                v_strain = torch.from_numpy(VFs_test[tag]['e'])[sbvf_idx]
                v_strain = torch.reshape(v_strain,(n_vfs,pred.shape[0],pred.shape[1]))
                #-----------------------------------------------------------------------------------------

                int_work = torch.sum(torch.reshape((pred * v_strain * area * ELEM_THICK),[n_vfs,t_pts,n_elems,3]),-1)

                int_work_real = torch.sum(torch.reshape((y_test * v_strain * area * ELEM_THICK),[n_vfs,t_pts,n_elems,3]),-1)

                f = f_test[:,:2][::n_elems,:]

                ext_work = torch.sum(torch.reshape(f,[t_pts,1,2])*v_disp,-1)

                # Computing losses
                test_loss = loss_fn(int_work, ext_work)
                test_losses[batch] = test_loss

                for i,(n,j) in enumerate(tuple(zip(tags,incs))):
                    #VFs[n]['u'][j] = v_u[:,i].detach().numpy()
                    #VFs[n]['e'][j] = torch.reshape(v_strain,[n_vfs,t_pts,n_elems,3])[:,i].detach().numpy()
                    w_virt[n]['w_int'][j] = torch.sum(torch.sum(int_work[:,i].detach(),-1)).numpy()
                    w_virt[n]['w_int_real'][j] = torch.sum(torch.sum(int_work_real[:,i].detach(),1)).numpy()
                    w_virt[n]['w_ext'][j] = torch.sum(torch.sum(ext_work[:,i].detach(),-1)).numpy()

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

    #partition['train'] = data[data['tag']=='m80_b80_x'].index.tolist()
    train_generator = DataGenerator(X, y, f, coord, info, partition["train"], batch_size, shuffle=False, std=True, t_pts=T_PTS)
    sbvf_generator = DataGenerator(X, y, f, coord, info, partition["train"], batch_size, shuffle=False, std=True, t_pts=T_PTS)

    test_generator = DataGenerator(X, y, f, coord, info, partition['test'], batch_size, shuffle=False, std=False, t_pts=T_PTS, scaler=train_generator.scaler_x)

    # Model variables
    N_INPUTS = X.shape[1]
    N_OUTPUTS = y.shape[1]

    N_UNITS = [18,12]
    H_LAYERS = len(N_UNITS)

    INCREMENTAL_VFS = False

    model_1 = NeuralNetwork(N_INPUTS, N_OUTPUTS, N_UNITS, H_LAYERS)
    #model_1 = InputConvexNN(N_INPUTS, N_OUTPUTS, N_UNITS, H_LAYERS)

    model_1.apply(init_weights)

    # geotorch.symmetric(model_1.layers[-1], "weight")

    # Training variables
    epochs = 5000

    # Optimization variables
    learning_rate = 0.1
    lr_mult = 0.5
    loss_fn = sbvf_loss
    mse = torch.nn.MSELoss()

    # layer_names = []
    # for n,p in model_1.named_parameters():
    #     layer_names.append(n)

    # layer_names.reverse()

    # parameters = []
    # prev_group_name = '.'.join(layer_names[0].split('.')[:2])

    # # store params & learning rates
    # for idx, name in enumerate(layer_names):

    #     # parameter group name
    #     cur_group_name = '.'.join(name.split('.')[:2])

    #     # update learning rate
    #     if cur_group_name != prev_group_name:
    #         learning_rate *= lr_mult
    #     prev_group_name = cur_group_name

    #     # display info
    #     print(f'{idx}: lr = {learning_rate:.6f}, {name}')

    #     # append layer parameters
    #     parameters += [{'params': [p for n, p in model_1.named_parameters() if n == name and p.requires_grad],
    #                     'lr':learning_rate}]

    optimizer = torch.optim.Adam(params=model_1.parameters(),lr=learning_rate)

    # lr_lambda = lambda x: math.exp(x * math.log(1e-7 / 1.0) / (epochs * len(train_generator)))
    #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50, cooldown=20, factor=0.95, min_lr=1e-5)
    #scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,base_lr=5e-3,max_lr=0.1,mode='exp_range',gamma=0.99994,cycle_momentum=False)
    constraints=weightConstraint(cond='plastic')

    # Container variables for history purposes
    train_loss = []
    l_stresses = []
    err_stresses = []
    v_work = []
    val_loss = []
    epochs_ = []
    # Initializing the early_stopping object
    #early_stopping = EarlyStopping(patience=1000, path='temp/checkpoint.pt', verbose=True)

    #wandb.watch(model_1)
    VFs_train = {key: {k: None for k in ['u','e','v_u']} for key in train_trials}
    VFs_test = {key: {k: None for k in ['u','e','v_u']} for key in test_trials}

    w_virt = {key: {k: dict.fromkeys(set(info['inc'])) for k in ['w_int','w_int_real','w_ext']} for key in set(info['tag'])}

    k_opt = 0

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
            vfs =  VFs_train[train_trials[0]]['e'].shape[0]
            sbvf_idx = random.sample(range(0, vfs), 20)
            sbvf_idx.sort()
        #--------------------------------------------------------------

        # Train loop
        start_train = time.time()

        #--------------------------------------------------------------
        batch_losses, l_stress, err_stress, batch_v_work, grad_norm = train_loop(train_generator, model_1, loss_fn, optimizer)
        #--------------------------------------------------------------

        q.put((w_virt[train_trials[0]],train_trials[0]))

        train_loss.append(torch.mean(batch_losses))
        l_stresses.append(torch.mean(l_stress))
        err_stresses.append(torch.mean(err_stress))
        v_work.append(torch.mean(batch_v_work))

        end_train = time.time()

        #Apply learning rate scheduling if defined
        try:
            scheduler.step(train_loss[t])
            print('. t_loss: %.6e -> lr: %.3e // [v_work] -> %.3e -- %.3fs' % (train_loss[t], scheduler._last_lr[0], v_work[t], end_train - start_train))
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
        q.put(([train_loss[t],val_loss[t],l_stresses[t],err_stresses[t]],'loss'))

        # if t > 200:
        #     if early_stopping.counter==0:
        #         w_virt_best = copy.deepcopy(w_virt)
        #         VFs_train_best = copy.deepcopy(VFs_train)
        #     # Check validation loss for early stopping
        #     early_stopping(val_loss[t], model_1)

        if t!= 0:
            tol_updt = (train_loss[0]/1.1**(k_opt))
            #tol_updt = (1e-1/1.5**k_opt)
            delta_loss = abs(train_loss[t]-train_loss[t-1])
            print('\n[%i] -- grad_norm (mean): %.6e | delta_loss: %.6e | tol_updt: %.6e' % (k_opt, grad_norm, delta_loss, tol_updt))

            if (delta_loss < tol_updt):
            #if (t%99==0):

                print('\nUpdating virtual fields [%i]' % (k_opt))

                get_sbvfs(copy.deepcopy(model_1), sbvf_generator)

                get_sbvfs(copy.deepcopy(model_1), test_generator, isTrain=False)

                #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50, cooldown=25, factor=0.2, min_lr=1e-5)
                k_opt += 1

        # if (t!= 0 and delta_loss < 1e-9) or early_stopping.early_stop:
        #     print("Early stopping")
        #     break

    q.put('Q')
    print("Done!")

    # load the last checkpoint with the best model
    #model_1.load_state_dict(torch.load('temp/checkpoint.pt'))

    epochs_ = np.reshape(np.array(epochs_), (len(epochs_),1))
    train_loss = np.reshape(np.array(train_loss), (len(train_loss),1))
    val_loss = np.reshape(np.array(val_loss), (len(val_loss),1))

    history = pd.DataFrame(np.concatenate([epochs_, train_loss, val_loss], axis=1), columns=['epoch','loss','val_loss'])


    task = r'[%i-%ix%i-%i]-%s-%i-VFs' % (N_INPUTS, N_UNITS[0], H_LAYERS, N_OUTPUTS, TRAIN_MULTI_DIR.split('/')[-2], count_parameters(model_1))

    output_task = 'outputs/' + TRAIN_MULTI_DIR.split('/')[-2] + '_sbvf_abs'
    output_loss = 'outputs/' + TRAIN_MULTI_DIR.split('/')[-2] + '_sbvf_abs/loss/'
    output_stats = 'outputs/' + TRAIN_MULTI_DIR.split('/')[-2] + '_sbvf_abs/stats/'
    output_models = 'outputs/' + TRAIN_MULTI_DIR.split('/')[-2] + '_sbvf_abs/models/'
    output_val = 'outputs/' + TRAIN_MULTI_DIR.split('/')[-2] + '_sbvf_abs/val/'
    output_logs = 'outputs/' + TRAIN_MULTI_DIR.split('/')[-2] + '_sbvf_abs/logs/'

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

    # # Reading mesh file
    # mesh, connectivity, dof = read_mesh(TRAIN_MULTI_DIR)

    # # Defining geometry limits
    # x_min = min(mesh[:,1])
    # x_max = max(mesh[:,1])
    # y_min = min(mesh[:,-1])
    # y_max = max(mesh[:,-1])

    # # Total degrees of freedom
    # total_dof = mesh.shape[0] * 2

    # # Defining edge boundary conditions
    # bcs = {
    #     'left': {
    #         'cond': [1,0],
    #         'dof': global_dof(mesh[mesh[:,1]==x_min][:,0])},
    #     'bottom': {
    #         'cond': [0,1],
    #         'dof': global_dof(mesh[mesh[:,-1]==y_min][:,0])},
    #     'right': {
    #         'cond': [2,0],
    #         'dof': global_dof(mesh[mesh[:,1]==x_max][:,0])},
    #     'top': {
    #         'cond': [0,0],
    #         'dof': global_dof(mesh[mesh[:,-1]==y_max][:,0])}
    # }

    # # Constructing element properties based on mesh info
    # elements = [Element(connectivity[i,:],mesh[connectivity[i,1:]-1,1:],dof[i,:]) for i in tqdm(range(connectivity.shape[0]))]

    # # Assembling global strain-displacement matrices
    # b_glob, b_bar, b_inv, active_dof = global_strain_disp(elements, total_dof, bcs)

    # # Loading data
    # df_list, _ = load_dataframes(TRAIN_MULTI_DIR)

    # # Merging training data
    # data = pd.concat(df_list, axis=0, ignore_index=True)

    # T_PTS = len(set(data['t']))

    # # Performing test/train split
    # partition = {"train": None, "test": None}

    # # if T_PTS==DATA_SAMPLES:
    # # Reorganizing dataset by tag, subsequent grouping by time increment
    # data_by_tag = [df for _, df in data.groupby(['tag'])]
    # random.shuffle(data_by_tag)
    # data_by_t = [[df for _, df in group.groupby(['t'])] for group in data_by_tag]
    # #random.shuffle(data_by_t)
    # data_by_batches = list(itertools.chain(*data_by_t))
    # #random.shuffle(data_by_batches)

    # batch_size = len(data_by_batches[0]) * T_PTS

    # data = pd.concat(data_by_batches).reset_index(drop=True)

    # trials = list(set(data['tag'].values))
    # test_trials = random.sample(trials, math.floor(len(trials)*TEST_SIZE))
    # train_trials = list(set(trials).difference(test_trials))

    # partition['train'] = data[data['tag'].isin(train_trials)].index.tolist()
    # partition['test'] = data[data['tag'].isin(test_trials)].index.tolist()

    # # Selecting model features
    # X, y, f, coord, info = select_features_multi(data)

    # # Preparing data generators for mini-batch training

    # #partition['train'] = data[data['tag']=='m80_b80_x'].index.tolist()
    # train_generator = DataGenerator(X, y, f, coord, info, partition["train"], batch_size, shuffle=False, std=True, t_pts=T_PTS)
    # test_generator = DataGenerator(X, y, f, coord, info, partition['test'], batch_size, shuffle=False, std=False, t_pts=T_PTS)

    # # Model variables
    # N_INPUTS = X.shape[1]
    # N_OUTPUTS = y.shape[1]

    # N_UNITS = [12,6]
    # H_LAYERS = len(N_UNITS)

    # model_1 = NeuralNetwork(N_INPUTS, N_OUTPUTS, N_UNITS, H_LAYERS)

    # model_1.apply(init_weights)

    # # Training variables
    # epochs = 50000

    # # Optimization variables
    # learning_rate = 0.1
    # loss_fn = sbvf_loss
    # f_loss = torch.nn.MSELoss()

    # optimizer = torch.optim.Adam(params=list(model_1.parameters()), lr=learning_rate)
    # #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50, factor=0.25, min_lr=1e-5)
    # #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=20, T_mult=1, eta_min=0.001)

    # #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,0.99)

    # #constraints=weightConstraint(cond='plastic')

    # # Container variables for history purposes
    # train_loss = []
    # v_work = []
    # val_loss = []
    # epochs_ = []
    # # Initializing the early_stopping object
    # early_stopping = EarlyStopping(patience=2000, path='temp/checkpoint.pt', verbose=True)

    # #wandb.watch(model_1)
    # VFs_train = {key: {k: dict.fromkeys(set(info['inc'])) for k in ['u','e','v_u']} for key in train_trials}
    # VFs_test = {key: {k: dict.fromkeys(set(info['inc'])) for k in ['u','e','v_u']} for key in test_trials}

    # w_virt = {key: {k: dict.fromkeys(set(info['inc'])) for k in ['w_int','w_int_real','w_ext']} for key in set(info['tag'])}

    # upd_tol = 1e-1
    # upd_tol_tune = 3.1
    # vf_updt = False
    # corr_factor = 0
    # isTraining = False

    # for t in range(epochs):

    #     print('\r--------------------\nEpoch [%d/%d]' % (t + 1, epochs))

    #     epochs_.append(t+1)

    #     #start_epoch = time.time()

    #     #Shuffling batches
    #     train_generator.on_epoch_end()
    #     test_generator.on_epoch_end()

    #     #--------------------------------------------------------------
    #     #Calculating VFs
    #     if t==0:
    #         print('Computing virtual fields...')
    #         get_sbvfs(model_1, train_generator)
    #         get_sbvfs(model_1, test_generator, isTrain=False)

    #     #--------------------------------------------------------------

    #     # Train loop
    #     start_train = time.time()

    #     #--------------------------------------------------------------
    #     batch_losses, batch_v_work = train_loop(train_generator, model_1, loss_fn, optimizer)
    #     #--------------------------------------------------------------

    #     #batch_losses, batch_v_work, v_disp, v_strain, v_u = train_loop(train_generator, model_1, loss_fn, optimizer)

    #     train_loss.append(torch.mean(batch_losses).item())
    #     v_work.append(torch.mean(batch_v_work).item())

    #     end_train = time.time()

    #     #Apply learning rate scheduling if defined
    #     try:
    #         scheduler.step(train_loss[t])
    #         #scheduler.step()
    #         print('. t_loss: %.6e -> lr: %.3e // [v_work] -> %.3e -- %.3fs' % (train_loss[t], scheduler._last_lr[0], v_work[t], end_train - start_train))
    #     except:
    #         print('. t_loss: %.6e // [v_work] -> %.3e -- %.3fs' % (train_loss[t], v_work[t], end_train - start_train))

    #     # Test loop
    #     start_test = time.time()

    #     # batch_val_losses = test_loop(test_generator, model_1, loss_fn, v_disp, v_strain)

    #     #-----------------------------------------------------------------------------------
    #     batch_val_losses = test_loop(test_generator, model_1, loss_fn)
    #     #-----------------------------------------------------------------------------------

    #     val_loss.append(torch.mean(batch_val_losses).item())

    #     end_test = time.time()

    #     print('. v_loss: %.6e -- %.3fs' % (val_loss[t], end_test - start_test))

    #     # end_epoch = time.time()

    #     if t > 100:
    #         # Check validation loss for early stopping
    #         early_stopping(val_loss[t], model_1)

    #     optimality = abs(train_loss[t]-train_loss[t-1])
    #     if t!= 0 and t % 10 == 0:
    #         print('\nComputing new virtual fields -- optimality: %.6e' % (optimality))
    #         get_sbvfs(model_1, train_generator)
    #         get_sbvfs(model_1, test_generator,isTrain=False)
    #     else:
    #         print('\noptimality: %.6e' % (optimality))
    #     if (t!= 0 and optimality < 1e-9) or early_stopping.early_stop:
    #             print("Early stopping")
    #             break

    # print("Done!")

    # # load the last checkpoint with the best model
    # #model_1.load_state_dict(torch.load('temp/checkpoint.pt'))

    # epochs_ = np.reshape(np.array(epochs_), (len(epochs_),1))
    # train_loss = np.reshape(np.array(train_loss), (len(train_loss),1))
    # val_loss = np.reshape(np.array(val_loss), (len(val_loss),1))

    # history = pd.DataFrame(np.concatenate([epochs_, train_loss, val_loss], axis=1), columns=['epoch','loss','val_loss'])


    # task = r'[%i-%ix%i-%i]-%s-%i-VFs' % (N_INPUTS, N_UNITS[0], H_LAYERS, N_OUTPUTS, TRAIN_MULTI_DIR.split('/')[-2], count_parameters(model_1))

    # output_task = 'outputs/' + TRAIN_MULTI_DIR.split('/')[-2] + '_sbvf_abs'
    # output_loss = 'outputs/' + TRAIN_MULTI_DIR.split('/')[-2] + '_sbvf_abs/loss/'
    # output_stats = 'outputs/' + TRAIN_MULTI_DIR.split('/')[-2] + '_sbvf_abs/stats/'
    # output_models = 'outputs/' + TRAIN_MULTI_DIR.split('/')[-2] + '_sbvf_abs/models/'
    # output_val = 'outputs/' + TRAIN_MULTI_DIR.split('/')[-2] + '_sbvf_abs/val/'
    # output_logs = 'outputs/' + TRAIN_MULTI_DIR.split('/')[-2] + '_sbvf_abs/logs/'

    # directories = [output_task, output_loss, output_stats, output_models, output_val, output_logs]

    # for dir in directories:
    #     try:
    #         os.makedirs(dir)

    #     except FileExistsError:
    #         pass

    # history.to_csv(output_loss + task + '.csv', sep=',', encoding='utf-8', header='true')

    # plot_history(history, output_loss, True, task)

    # torch.save(model_1.state_dict(), output_models + task + '.pt')
    # np.save(output_logs + 'sbvfs.npy', VFs_train)
    # np.save(output_logs + 'w_virt.npy', w_virt)
    # #joblib.dump([VFs, W_virt], 'sbvfs.pkl')
    # if train_generator.std == True:
    #     joblib.dump(train_generator.scaler_x, output_models + task + '-scaler_x.pkl')

    # Deleting temp folder
    try:
        shutil.rmtree('temp/')
    except FileNotFoundError:
        pass

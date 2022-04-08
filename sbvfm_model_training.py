# ---------------------------------
#    Library and function imports
# ---------------------------------
import re
import operator
import joblib
from sklearn.utils import shuffle
from constants import *
from functions import (
    custom_loss, 
    load_dataframes, 
    prescribe_u, 
    select_features_multi, 
    standardize_data, 
    plot_history, 
    read_mesh,
    global_strain_disp,
    param_deltas,
    global_dof)
from functions import (EarlyStopping, NeuralNetwork, Element)
from sklearn.model_selection import GroupShuffleSplit
import copy
from re import S
from torch import nn
import torch.nn.functional as F

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

from sympy import pi
from tqdm import tqdm

from torch.nn.utils import (
  parameters_to_vector as Params2Vec,
  vector_to_parameters as Vec2Params
)
# -------------------------------
#        Class definitions
# -------------------------------
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, deformation, stress, force, coord, list_IDs, batch_size, shuffle, std=True, t_pts=1):
        super().__init__()
        self.X = deformation
        self.y = stress
        self.f = force
        self.coord = coord[['dir','id','cent_x','cent_y','area']]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.list_IDs = list_IDs
        self.std = std
        self.t_pts = t_pts
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
            if batch_size > 9:
                index_groups = np.array_split(indexes, self.t_pts)
                permuted = [np.random.permutation(index_groups[i]) for i in range(len(index_groups))]
                indexes = np.hstack(permuted)
                X, y, f, coord = self.__data_generation(indexes)
            else:
                X, y, f, coord = self.__data_generation(np.random.permutation(indexes))
        else:
            X, y, f, coord = self.__data_generation(indexes)
        
        return X, y, f, coord

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(0, len(self.list_IDs), self.batch_size)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
 
    def standardize(self):
        'Standardizes neural network input data'
        idx = self.X.index
        self.X, _, _, self.scaler_x, _, _ = standardize_data(self.X, self.y, self.f)

        self.X = pd.DataFrame(self.X, index=idx)
        
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        X = np.asarray(self.X.iloc[list_IDs_temp], dtype=np.float32)
        y = np.asarray(self.y.iloc[list_IDs_temp], dtype=np.float32)
        f = np.asarray(self.f.iloc[list_IDs_temp], dtype=np.float32)
        coord = np.asarray(self.coord.iloc[list_IDs_temp], dtype=np.float32)
        return X, y, f, coord



# class NeuralNetwork(nn.Module):
#     def __init__(self, input_size, output_size, hidden_size, n_hidden_layers=1):
#         super(NeuralNetwork, self).__init__()
#         self.input_size = input_size
#         self.hidden_size  = hidden_size
#         self.n_hidden_layers = n_hidden_layers
#         self.output_size = output_size

#         self.layers = nn.ModuleList()

#         for i in range(self.n_hidden_layers):
#             if i == 0:
#                 in_ = self.input_size
#             else:
#                 in_ = self.hidden_size

#             self.layers.append(torch.nn.Linear(in_, self.hidden_size, bias=True))

#         self.layers.append(torch.nn.Linear(self.hidden_size, self.output_size, bias=True))

#         self.activation_h = torch.nn.PReLU(self.hidden_size)
#         self.activation_o = torch.nn.PReLU(self.output_size)

#     def forward(self, x):

#         for layer in self.layers[:-1]:
            
#             x = self.activation_h(layer(x))
            
#         #return self.layers[-1](x)
#         return self.activation_o(self.layers[-1](x))

# -------------------------------
#       Method definitions
# -------------------------------

def batch_jacobian(f, x):
    f_sum = lambda x: torch.sum(f(x), axis=0)
    return jacobian(f_sum, x,create_graph=True).permute(1,0,2)


def sigma_deltas(model, X_train, pred, param_deltas):

    model_eval = copy.deepcopy(model)
    
    delta_stress = torch.zeros(size=(len(param_deltas), pred.shape[0], pred.shape[1]))

    for i,param_dict in enumerate(param_deltas):
        with torch.no_grad():
            model_eval.eval()
            model_eval.load_state_dict(param_dict)
            pred_eval = model_eval(X_train)
            delta_stress[i,:,:] += pred - pred_eval

    return delta_stress

def sbv_fields(d_sigma, id, b_glob, t_pts, n_elems):

    id_reshaped = torch.reshape(id, (t_pts,n_elems))

    sorted_ids, indices = torch.sort(id_reshaped,-1)
    
    d_sigma_reshaped = torch.reshape(d_sigma,(d_sigma.shape[0],t_pts,n_elems*3,1))

    #d_sigma_ordered = torch.zeros_like(d_sigma_reshaped)

    # for i in range(d_sigma_ordered.shape[0]):
    #     for j in range(d_sigma_ordered.shape[1]):
    #         d_sigma_ordered[i,j] = d_sigma_reshaped[i][j][:,indices[j,:]]

    b = b_glob.unsqueeze(0).repeat(d_sigma.shape[0],t_pts,n_elems*3,1)

    v_disp = torch.linalg.pinv(b_glob) @ d_sigma_reshaped

    # left_symm = global_dof(mesh[mesh[:,1]==0][:,0])[::2]
    # bottom_symm = global_dof(mesh[mesh[:,-1]==0][:,0])[1::2]

    # b_glob = prescribe_u(b_glob,[left_symm,bottom_symm])

    v_strain = b_glob @ v_disp

    print('hey')

    
    return None

def init_weights(m):
    '''
    Performs the weight initialization of a neural network

    Parameters
    ----------
    m : NeuralNetwork object
        Neural network model, instance of NeuralNework class
    '''
    if isinstance(m, nn.Linear) and (m.bias != None):
        torch.nn.init.kaiming_normal_(m.weight)
        #torch.nn.init.xavier_normal(m.weight)
        #torch.nn.init.zeros_(m.bias)
        torch.nn.init.ones_(m.bias)
   

def train_loop(dataloader, model, loss_fn, optimizer, param_deltas):
    '''
    Custom loop for neural network training, using mini-batches

    '''
    num_batches = len(dataloader)
    losses = torch.zeros(num_batches)
    l_sec = torch.zeros(num_batches)
    v_work_real = torch.zeros(num_batches)
    t_pts = dataloader.t_pts
    n_elems = batch_size//t_pts

    for batch in range(num_batches):
        # Extracting variables for training
        X_train, y_train, f_train, coord = dataloader[batch]
        
        # Converting to pytorch tensors
        X_train = torch.tensor(X_train, dtype=torch.float64)
        X_train.requires_grad = True
        y_train = torch.tensor(y_train, dtype=torch.float64)
        f_train = torch.tensor(f_train, dtype=torch.float64) + 0.0
        coord_torch = torch.tensor(coord, dtype=torch.float64)
         
        # Extracting centroid coordinates
        #dir = coord_torch[:,0][0]
        id = coord_torch[:,1]
        
        #cent_x = coord_torch[:,2]
        #cent_y = coord_torch[:,3]
        
        area = torch.reshape(coord_torch[:,4],[batch_size,1])

        # Defining surface coordinates where load is applied
        #n_surf_nodes = int((9**0.5) + 1)
     
        #x = LENGTH * torch.ones(n_surf_nodes, dtype=torch.float)
        #y = torch.tensor(np.linspace(0, LENGTH, n_surf_nodes), dtype=torch.float)

        # Computing prediction
        # pred_1 = model[0](X_train[:,[0,3,6]])
        # pred_2 = model[1](X_train[:,[1,4,7]])
        # pred_3 = model[2](X_train[:,[2,5,8]])

        #pred = torch.cat([pred_1,pred_2,pred_3],1)
        pred=model(X_train)

        d_sigma = sigma_deltas(model, X_train, pred, param_deltas)

        total_vfs, v_disp, v_strain = sbv_fields(d_sigma, id, b_glob, t_pts, n_elems)

        # Calculating global force from stress predictions
        # global_f_pred = torch.sum(pred * ELEM_AREA * ELEM_THICK / LENGTH, 0)[:-1]

        #dsde = batch_jacobian(model,X_train)
        #e_pred = torch.linalg.solve(dsde,pred)
        #L = torch.linalg.eigh(0.5*(dsde+torch.permute(dsde,(0,2,1)))).eigenvalues
        #SPD = torch.einsum('bij,bjk->bik', dsde, torch.permute(dsde,(0,2,1)))
        #C = torch.linalg.cholesky(SPD)
        #H = torch.einsum('bij,bjk->bik', dsde, torch.permute(dsde,(0,2,1)))
        # x = copy.deepcopy(X_train).detach().numpy()
        # x = torch.tensor(dataloader.scaler_x.inverse_transform(x))
        # dsde = get_batch_jacobian(model, x, N_OUTPUTS)[:,:,-3:]
        #s = torch.reshape(dsde @ torch.reshape(torch.tensor(x[:,3:]),[9,3,1]),[9,3])

        #e = torch.reshape(torch.linalg.solve(dsde, torch.reshape(pred,[9,3,1])),[9,3])
        # ltri = torch.stack([torch.tril(dsde[i]).flatten() for i in range(batch_size)])
        # ltri = torch.stack([ltri[i][ltri[i].nonzero()] for i in range(batch_size)]).detach().numpy()
        #print('hey')
        # #Ensure H
        # sols = []
        # for i in range(batch_size):
        #     l = fsolve(func, [1.0,0.0,1.0,0.0,0.0,1.0], args=ltri[i].reshape(-1), xtol=1e-3)
        #     h = np.zeros((3, 3))
        #     inds = np.triu_indices(len(h))
        #     h[inds] = l
        #     h = torch.tensor(h + h.T - np.diag(np.diag(h)), dtype=torch.float32)
        #     sols.append(h)

        # H = torch.stack(sols)
        # e = X_train[:,-3:].reshape([batch_size, 3, 1])
        # s = (H @ e).reshape([batch_size,3])
        
        # w_int = torch.reshape(-ELEM_THICK * torch.sum(pred * v_strain[:] * ELEM_AREA, -1, keepdims=True),[6,-1]).T
        # w_int_real =torch.reshape(-ELEM_THICK * torch.sum(y_train * v_strain[:] * ELEM_AREA, -1, keepdims=True),[6,-1]).T

        # w_int = torch.cat([coord[:,1:],w_int],1)
        # w_int_real = torch.cat([coord[:,1:],w_int_real],1)

        # Computing the internal virtual works
        #int_work = - torch.sum(torch.reshape(torch.sum(pred * v_strain[:] * ELEM_AREA * ELEM_THICK, -1, keepdims=True),[total_vfs,batch_size//batch_size,batch_size,1]),2)
        #int_work = torch.reshape(int_work,[-1,1])

        #int_work = -torch.sum(torch.sum(pred * v_strain[:] * area * ELEM_THICK, -1, keepdims=True),1)
        #int_work = -torch.sum(torch.sum(pred.view([t_pts,n_elems,3]) * v_strain.view([total_vfs,t_pts,n_elems,3])* area.view([t_pts,n_elems,1]) * ELEM_THICK, -1, keepdims=True),2)

        int_work = torch.reshape((pred * v_strain * area * ELEM_THICK),[total_vfs,t_pts,n_elems,3])
       
        # Computing the internal virtual work coming from stress labels (results checking only)
        # int_work_real = torch.sum(torch.reshape(torch.sum(y_train * v_strain[:] * ELEM_AREA * ELEM_THICK, -1, keepdims=True),[total_vfs,batch_size//batch_size,batch_size,1]),2)        
        # int_work_real = torch.reshape(int_work_real,[-1,1])
        
        #int_work_real = -torch.sum(torch.sum(y_train * v_strain[:] * area * ELEM_THICK, -1, keepdims=True),1) 
        #int_work_real = -torch.sum(torch.sum(y_train.view([t_pts,n_elems,3]) * v_strain.view([total_vfs,t_pts,n_elems,3])* area.view([t_pts,n_elems,1]) * ELEM_THICK, -1, keepdims=True),2)

        int_work_real = torch.reshape((y_train * v_strain * area * ELEM_THICK),[total_vfs,t_pts,n_elems,3])
        
        # Extracting global force components and filtering duplicate values
        #f = f_train[:,:2][::batch_size,:]
        f = f_train[:,:2][::n_elems,:]

        # Computing external virtual work
        #ext_work = torch.mean(torch.sum(f*v_disp[:],-1,keepdims=True),1)     
        ext_work = torch.sum(torch.reshape(f,[t_pts,1,2])*torch.mean(v_disp,1),-1,keepdim=True).permute(1,0,2)
        
        # Compute L1 loss component
        # l1_weight = 0.01
        # l1_parameters = []
        # for parameter in model.layers[0].parameters():
        #     l1_parameters.append(parameter.view(-1))
        # l = l1_weight * model.compute_l1_loss(torch.cat(l1_parameters))
        #l= torch.sum(torch.square(torch.abs(e_pred-X_train)))
        
        l=0

        # Computing losses        
        #loss = loss_fn(int_work, ext_work)
        loss = loss_fn(int_work,ext_work)
        cost = loss_fn(int_work_real, ext_work)

        # Backpropagation and weight's update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Saving loss values
        losses[batch] = loss
        v_work_real[batch] = cost
        l_sec[batch] = l

        print('\r>Train: %d/%d' % (batch + 1, num_batches), end='')
    
    
    return losses, v_work_real, total_vfs, l_sec

def test_loop(dataloader, model, loss_fn):
    
    num_batches = len(dataloader)
    test_losses = torch.zeros(num_batches)
    Wint = torch.zeros((num_batches,1,3))
    Wint_real = torch.zeros((num_batches,1,3))
    Wext = torch.zeros((num_batches,1,3))
    t_pts = dataloader.t_pts
    n_elems = batch_size//t_pts

    model.eval()
    with torch.no_grad():

        for batch in range(num_batches):

            # Extracting variables for testing
            X_test, y_test, f_test, coord = dataloader[batch]
            
            # Converting to pytorch tensors
            X_test = torch.tensor(train_generator.scaler_x.transform(X_test), dtype=torch.float64)
            #X_test = torch.tensor(X_test, dtype=torch.float64)
            y_test = torch.tensor(y_test, dtype=torch.float64)
            f_test = torch.tensor(f_test, dtype=torch.float64) + 0.0
            coord_torch = torch.tensor(coord, dtype=torch.float64)
            
            # # Extracting centroid coordinates
            dir = coord_torch[:,0][0]
            id = coord_torch[:,1]
            cent_x = coord_torch[:,2]
            cent_y = coord_torch[:,3]
            area = torch.reshape(coord_torch[:,4],[batch_size,1])

            # # Defining surface coordinates where load is applied
            n_surf_nodes = int((batch_size**0.5) + 1)
            x = LENGTH * torch.ones(n_surf_nodes, dtype=torch.float32)
            y = torch.tensor(np.linspace(0, LENGTH, n_surf_nodes), dtype=torch.float32)

            total_vfs, v_disp, v_strain = get_v_fields(cent_x, cent_y, x, y, dir)

            # pred_1 = model[0](X_test[:,[0,3,6]])
            # pred_2 = model[1](X_test[:,[1,4,7]])
            # pred_3 = model[2](X_test[:,[2,5,8]])

            # pred = torch.cat([pred_1,pred_2,pred_3],1)
            pred = model(X_test)

            # Computing the internal virtual works
            # int_work = -ELEM_THICK * torch.sum(torch.reshape(torch.sum(pred * v_strain[:] * ELEM_AREA, -1, keepdims=True),[total_vfs,batch_size//batch_size,batch_size,1]),2)
            # int_work = torch.reshape(int_work,[-1,1])
            #int_work = -torch.sum(torch.sum(pred * v_strain[:] * area[:] * ELEM_THICK, -1, keepdims=True),1)

            # int_work = -torch.sum(torch.sum(pred.view([t_pts,n_elems,3]) * v_strain.view([total_vfs,t_pts,n_elems,3])* area.view([t_pts,n_elems,1]) * ELEM_THICK, -1, keepdims=True),2)

            int_work = torch.reshape((pred * v_strain * area * ELEM_THICK),[total_vfs,t_pts,n_elems,3])

            # int_work_real = -ELEM_THICK * torch.sum(torch.reshape(torch.sum(y_test * v_strain[:] * ELEM_AREA, -1, keepdims=True),[total_vfs,batch_size//batch_size,batch_size,1]),2)
            # int_work_real = torch.reshape(int_work_real,[-1,1])
            #int_work_real = -torch.sum(torch.sum(y_test * v_strain[:] * area[:] * ELEM_THICK, -1, keepdims=True),1) 

            #int_work_real = -torch.sum(torch.sum(y_test.view([t_pts,n_elems,3]) * v_strain.view([total_vfs,t_pts,n_elems,3])* area.view([t_pts,n_elems,1]) * ELEM_THICK, -1, keepdims=True),2)

            int_work_real = torch.reshape((y_test * v_strain * area * ELEM_THICK),[total_vfs,t_pts,n_elems,3])

            # Extracting global force components and filtering duplicate values
            #f = f_test[:,:2][::batch_size,:]

            f = f_test[:,:2][::n_elems,:]
            
            # Computing external virtual work
            #ext_work = torch.mean(torch.sum(f*v_disp[:],-1,keepdims=True),1)
            
            ext_work = torch.sum(f.view(t_pts,1,2)*torch.mean(v_disp,1),-1,keepdims=True).permute(1,0,2) 

            # Computing losses        
            # test_loss = loss_fn(int_work, ext_work)
            test_loss = loss_fn(int_work, ext_work)
            test_losses[batch] = test_loss

            # Wint[batch] = torch.mean(int_work.T,1)
            # Wint_real[batch] = torch.mean(int_work_real.T,1)
            # Wext[batch] = torch.mean(ext_work.T,1)

            print('\r>Test: %d/%d' % (batch + 1, num_batches), end='')

    
    return test_losses#, torch.mean(Wint,0), torch.mean(Wint_real,0), torch.mean(Wext,0)

# -------------------------------
#           Main script
# -------------------------------

# Initiating wandb
#wandb.init(project="pytorch_linear_model", entity="rmbl")

torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=8)

# Specifying random seed
random.seed(SEED)

mesh, connectivity, dof = read_mesh(TRAIN_MULTI_DIR)

elements = [Element(connectivity[i,:],mesh[connectivity[i,1:]-1,1:],dof[i,:]-1) for i in tqdm(range(connectivity.shape[0]))]

b_glob = global_strain_disp(elements,mesh.shape[0])

# Loading data
df_list, _ = load_dataframes(TRAIN_MULTI_DIR)

# Merging training data
data = pd.concat(df_list, axis=0, ignore_index=True)

DATA_POINTS = len(df_list[0])

T_PTS = 8

# Performing test/train split
partition = {"train": None, "test": None}

if T_PTS == DATA_SAMPLES:
    # Reorganizing dataset by tag, subsequent grouping by time increment
    data_by_tag = [df for _, df in data.groupby(['tag'])]
    random.shuffle(data_by_tag)
    data_by_t = [[df for _, df in group.groupby(['t'])] for group in data_by_tag]
    #random.shuffle(data_by_t)
    data_by_batches = list(itertools.chain(*data_by_t))
    #random.shuffle(data_by_batches)

    data = pd.concat(data_by_batches).reset_index(drop=True)

    trials = list(set(data['tag'].values))
    test_trials = random.sample(trials, round(len(trials)*TEST_SIZE))
    train_trials = list(set(trials).difference(test_trials))

    partition['train'] = data[data['tag'].isin(train_trials)].index.tolist()
    partition['test'] = data[data['tag'].isin(test_trials)].index.tolist()

else:
    # Reorganizing dataset by time increment, subsequent grouping by tag and final shuffling
    data_by_t = [df for _, df in data.groupby(['t'])]
    random.shuffle(data_by_t)
    data_by_tag = [[df for _, df in group.groupby(['tag'])] for group in data_by_t]
    random.shuffle(data_by_tag)
    data_by_batches = list(itertools.chain(*data_by_tag))
    random.shuffle(data_by_batches)

    data = pd.concat(data_by_batches).reset_index(drop=True)

    partition['train'], partition['test'] = next(GroupShuffleSplit(test_size=TEST_SIZE, n_splits=2, random_state = SEED).split(data, groups=data['t']))

batch_size = len(data_by_batches[0]) * T_PTS

#Concatenating data groups
#data = pd.concat(data_by_batches).reset_index(drop=True)

# data = pd.concat(data_by_t).reset_index(drop=True)

# m = [0,10,20,30]
# b = [270,300,330]

# from itertools import product
# import matplotlib.pyplot as plt
# trials = list(product(m, b))

# for trial in trials:

#     elem_ids = list(set(data['id']))
#     lines = dict.fromkeys(elem_ids)

#     df_trial = data[data['tag']==('m%i_b%i_x') % (trial[0],trial[1])]

#     for elem in elem_ids:
#         lines[elem] = df_trial[df_trial['id']==elem][['exx_t','eyy_t','exy_t','sxx_t','syy_t','sxy_t','fxx_t']]

#     fig, axs = plt.subplots(1,3)
#     fig.suptitle(('m%i_b%i_x') % (trial[0],trial[1]))
#     fig.set_size_inches(19.2,10.8,forward=True)
#     for key, line in lines.items():
#         axs[0].plot(line['exx_t'],line['sxx_t'], label=('id-%i') % key)
#         axs[0].set_xlabel('e_xx')
#         axs[0].set_ylabel('s_xx[MPa]')
#         axs[1].plot(line['eyy_t'],line['syy_t'], label=('id-%i') % key)
#         axs[1].set_xlabel('e_yy')
#         axs[1].set_ylabel('s_yy[MPa]')
#         axs[2].plot(line['exy_t'],line['sxy_t'], label=('id-%i') % key)
#         axs[2].set_xlabel('e_xy')
#         axs[2].set_ylabel('s_xy[MPa]')
#     # fig, axs = plt.subplots(2,3)
#     # fig.suptitle(('m%i_b%i_x') % (trial[0],trial[1]))
#     # fig.set_size_inches(19.2,10.8,forward=True)
#     # for key, line in lines.items():
        
#     #     axs[0,0].plot(line['fxx_t'],line['exx_t'], label=('id-%i') % key)
#     #     axs[0,0].set_xlabel('F[N]')
#     #     axs[0,0].set_ylabel('e_xx[%]')
#     #     axs[0,1].plot(line['fxx_t'],line['eyy_t'], label=('id-%i') % key)
#     #     axs[0,1].set_xlabel('F[N]')
#     #     axs[0,1].set_ylabel('e_yy[%]')
#     #     axs[0,2].plot(line['fxx_t'],line['exy_t'], label=('id-%i') % key)
#     #     axs[0,2].set_xlabel('F[N]')
#     #     axs[0,2].set_ylabel('e_xy[%]')
#     #     axs[1,0].plot(line['fxx_t'],line['sxx_t'], label=('id-%i') % key)
#     #     axs[1,0].set_xlabel('F[N]')
#     #     axs[1,0].set_ylabel('s_xx[MPa]')
#     #     axs[1,1].plot(line['fxx_t'],line['syy_t'], label=('id-%i') % key)
#     #     axs[1,1].set_xlabel('F[N]')
#     #     axs[1,1].set_ylabel('s_yy[MPa]')
#     #     axs[1,2].plot(line['fxx_t'],line['sxy_t'], label=('id-%i') % key)
#     #     axs[1,2].set_xlabel('F[N]')
#     #     axs[1,2].set_ylabel('s_xy[MPa]')

#     plt.legend(loc='best')
#     #plt.show()
#     plt.savefig(('m%i_b%i_x_scurve.png') % (trial[0],trial[1]), dpi=100, bbox_inches='tight', format='png')

# Selecting model features
X, y, f, coord = select_features_multi(data)

# Preparing data generators for mini-batch training
train_generator = DataGenerator(X, y, f, coord, partition["train"], batch_size, True, std=True, t_pts=T_PTS)
test_generator = DataGenerator(X, y, f, coord, partition['test'], batch_size, False, std=False, t_pts=T_PTS)

# Model variables
N_INPUTS = X.shape[1]
N_OUTPUTS = y.shape[1]

N_UNITS = 6
H_LAYERS = 1

model_1 = NeuralNetwork(N_INPUTS, N_OUTPUTS, N_UNITS, H_LAYERS)

model_1.apply(init_weights)

# Training variables
epochs = 500

# Optimization variables
learning_rate = 0.01
loss_fn = custom_loss
f_loss = torch.nn.MSELoss()

optimizer = torch.optim.Adam(params=list(model_1.parameters()), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=20, factor=0.2, threshold=1e-3, min_lr=1e-5)

# Container variables for history purposes
train_loss = []
v_work = []
val_loss = []
epochs_ = []
l = []
# Initializing the early_stopping object
#early_stopping = EarlyStopping(patience=12, verbose=True)

#wandb.watch(model_1)

w_int = np.zeros((epochs,1,4))
w_int_real = np.zeros((epochs,1,4))
w_ext = np.zeros((epochs,1,4))

grads = []

for t in range(epochs):

    print('\r--------------------\nEpoch [%d/%d]' % (t + 1, epochs))

    epochs_.append(t+1)
    
    start_epoch = time.time()

    #Shuffling batches
    train_generator.on_epoch_end()

    param_dicts = param_deltas(model_1)
    
    # Train loop
    start_train = time.time()
    batch_losses, batch_v_work, n_vfs, l_sec = train_loop(train_generator, model_1, loss_fn, optimizer, param_dicts)
    
    train_loss.append(torch.mean(batch_losses).item())
    v_work.append(torch.mean(batch_v_work).item())
    l.append(torch.mean(l_sec).item())

    end_train = time.time()
    
    #Apply learning rate scheduling if defined
    try:
        scheduler.step(train_loss[t])       
        print('. t_loss: %.3e | l_sec: %.3e -> lr: %.3e // [v_work] -> %.3e -- %.3fs' % (train_loss[t], l[t], scheduler._last_lr[0], v_work[t], end_train - start_train))
    except:
        print('. t_loss: %.3e | l_sec: %.3e // [v_work] -> %.3e -- %.3fs' % (train_loss[t], l[t], v_work[t], end_train - start_train))

    # Test loop
    start_test = time.time()

    batch_val_losses = test_loop(test_generator, model_1, loss_fn)

    val_loss.append(torch.mean(batch_val_losses).item())

    end_test = time.time()

    # w_int[t] = np.insert(a.detach().numpy(), 0, t, axis=1)
    # w_int_real[t] = np.insert(b.detach().numpy(), 0, t, axis=1)
    # w_ext[t] = np.insert(c.detach().numpy(), 0, t, axis=1)

    print('. v_loss: %.3e -- %.3fs' % (val_loss[t], end_test - start_test))

    end_epoch = time.time()

    # # Check validation loss for early stopping
    # early_stopping(val_loss[t], model)

    # if early_stopping.early_stop:
    #      print("Early stopping")
    #      break

    # wandb.log({
    #     "Epoch": t,
    #     "Train Loss": train_loss[t],
    #     "Test Loss": val_loss[t],
    #     #"Learning Rate": scheduler._last_lr[0]
    #     }
    #     )

print("Done!")

# load the last checkpoint with the best model
#model.load_state_dict(torch.load('checkpoint.pt'))

# w_int = pd.DataFrame(np.reshape(w_int,(-1,4)),columns=['epoch','vf1','vf2','vf3'])
# w_int_real = pd.DataFrame(np.reshape(w_int_real,(-1,4)),columns=['epoch','vf1','vf2','vf3'])
# w_ext = pd.DataFrame(np.reshape(w_ext,(-1,4)),columns=['epoch','vf1','vf2','vf3'])
# joblib.dump(w_int,'w_int.pkl')
# joblib.dump(w_ext,'w_ext.pkl')
# joblib.dump(w_int_real,'w_int_real.pkl')

epochs_ = np.reshape(np.array(epochs_), (len(epochs_),1))
train_loss = np.reshape(np.array(train_loss), (len(train_loss),1))
val_loss = np.reshape(np.array(val_loss), (len(val_loss),1))

history = pd.DataFrame(np.concatenate([epochs_, train_loss, val_loss], axis=1), columns=['epoch','loss','val_loss'])


task = r'[%i-%ix%i-%i]-%s-%i-VFs' % (N_INPUTS, N_UNITS, H_LAYERS, N_OUTPUTS, TRAIN_MULTI_DIR.split('/')[-2], n_vfs)

import os
output_task = 'outputs/' + TRAIN_MULTI_DIR.split('/')[-2] + '_indirect'
output_loss = 'outputs/' + TRAIN_MULTI_DIR.split('/')[-2] + '_indirect/loss/'
output_prints = 'outputs/' + TRAIN_MULTI_DIR.split('/')[-2] + '_indirect/prints/'
output_models = 'outputs/' + TRAIN_MULTI_DIR.split('/')[-2] + '_indirect/models/'
output_val = 'outputs/' + TRAIN_MULTI_DIR.split('/')[-2] + '_indirect/val/'

directories = [output_task, output_loss, output_prints, output_models, output_val]

for dir in directories:
    try:
        os.makedirs(dir)  
        
    except FileExistsError:
        pass

history.to_csv(output_loss + task + '.csv', sep=',', encoding='utf-8', header='true')

plot_history(history, output_prints, True, task)

torch.save(model_1.state_dict(), output_models + task + '_1.pt')
if train_generator.std == True:
    joblib.dump(train_generator.scaler_x, output_models + task + '-scaler_x.pkl')
#joblib.dump(stress_scaler, output_models + task + '-scaler_stress.pkl')
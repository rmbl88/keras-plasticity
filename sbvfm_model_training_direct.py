# ---------------------------------
#    Library and function imports
# ---------------------------------
from audioop import mul
import cProfile, pstats
from doctest import master
import os
from pyexpat import model
import shutil
import joblib
from sklearn import model_selection
from constants import *
from functions import (
    InputConvexNN,
    layer_wise_lr,
    rotate_tensor,
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
from torch import cumulative_trapezoid, nn
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
from sklearn import preprocessing
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.ticker import MaxNLocator
import multiprocessing
import time
import random
from tkinter import *
import geotorch
import matplotlib.pyplot as plt
import gc
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
    def __init__(self, deformation, stress, force, coord, info, list_IDs, batch_sizes, shuffle=False, std=True, t_pts=1, scaler=None):
        super().__init__()

        self.X = deformation.iloc[list_IDs].reset_index(drop=True)
        self.y = stress.iloc[list_IDs].reset_index(drop=True)
        self.f = force.iloc[list_IDs].reset_index(drop=True)
        self.coord = coord[['id','area']].iloc[list_IDs].reset_index(drop=True)
        self.tag = info['tag'].iloc[list_IDs].reset_index(drop=True)
        self.t = info[['inc','t','theta_p','exx_t','eyy_t','exy_t','sxx_t','syy_t','sxy_t']].iloc[list_IDs].reset_index(drop=True)
        self.batch_sizes = batch_sizes
        self.shuffle = shuffle
        self.list_IDs = list_IDs
        self.std = std
        self.t_pts = t_pts
        self.scaler_x = scaler

        if self.std == True:
            self.standardize()

        self.on_epoch_end()

    def __len__(self):
        return len(self.batch_sizes)
        #return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        #indexes = np.array([self.indexes[index]+i for i in range(self.batch_size)])  # Generate indexes of the batch
        indexes = np.array([self.indexes[index]+i for i in range(self.batch_sizes[self.tags[index]])])

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
        #self.indexes = np.arange(0, len(self.list_IDs), self.batch_size)
        #if self.shuffle == True:
        self.tags = list(set(self.tag))
        np.random.shuffle(self.tags)
        self.indexes = np.array([self.tag[self.tag==self.tags[i]].index[0] for i in range(len(self.tags))])
        #print('hey')

    def standardize(self):
        'Standardizes neural network input data'
        idx = self.X.index
        self.X, _, _, self.scaler_x, _, _ = standardize_data(self.X, self.y, self.f)

        self.X = pd.DataFrame(self.X, index=idx)
        #self.y = pd.DataFrame(self.y, index=idx)

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
#       Method definitions
# -------------------------------

def batch_jacobian(f, x):
    f_sum = lambda x: torch.sum(f(x), axis=0)
    return jacobian(f_sum, x,create_graph=True).permute(1,0,2)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train():

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
            torch.nn.init.kaiming_uniform_(m.weight) #RELU
            #torch.nn.init.xavier_normal(m.weight)
            #torch.nn.init.zeros_(m.bias)
            #torch.nn.init.ones_(m.bias)
            m.bias.data.fill_(0.0)

    def train_loop(dataloader, model, loss_fn, optimizer):
        '''
        Custom loop for neural network training, using mini-batches

        '''

        num_batches = len(dataloader)
        losses = torch.zeros(num_batches)
        
        g_norm = []

        l0_loss = torch.zeros_like(losses) 
        l1_loss = torch.zeros_like(losses)
        wp_loss = torch.zeros_like(losses)
        ch_loss = torch.zeros_like(losses)
        I1_loss = torch.zeros_like(losses)

        # t_pts = dataloader.t_pts
        # n_elems = batch_size // t_pts
               
        model.train()
        optimizer.zero_grad(set_to_none=True)
        
        for batch in range(num_batches):

            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=USE_AMP):

                # Extracting variables for training
                X_train, y_train, _, _, _, inc = dataloader[batch]
                
                t_pts = dataloader.t_pts[dataloader.tags[batch]]
                n_elems = dataloader.batch_sizes[dataloader.tags[batch]] // t_pts

                # Converting to pytorch tensors
                X_train = torch.from_numpy(X_train).to(device)
                #X_train = X_train[~torch.any(X_train.isnan(),dim=1)]
                y_train = torch.from_numpy(y_train).to(device)
                #y_train = y_train[:-n_elems]
                # Importing principal angles
                theta_p = torch.from_numpy(inc[['theta_p']].values).reshape(-1).to(device)
                # theta_p = theta_p[~torch.any(theta_p.isnan(),dim=1)]
                # theta_p = torch.cat((torch.zeros(n_elems,1),theta_p),0)
                #eps = torch.from_numpy(inc[['exx_t','eyy_t','exy_t','sxx_t','syy_t','sxy_t']].values)
                t = torch.from_numpy(inc['t'].values).reshape(-1,1).to(device)
                #dt = torch.diff(t,dim=0).reshape(-1,1)

                pred = model(X_train) # stress rate
                #pred = torch.cat((torch.zeros(n_elems,pred.shape[-1]),pred),0)
                #pred = torch.cumulative_trapezoid(pred.reshape(t_pts,n_elems,pred.shape[-1]),t.reshape(t_pts,n_elems,t.shape[-1]),dim=0).reshape(-1,pred.shape[-1])     # stress
                pred *= t.repeat(1,pred.shape[-1])
                # pred = torch.cumsum(pred.reshape(t_pts-1,n_elems,pred.shape[-1]),0).reshape(-1,pred.shape[-1])
        
                tril_indices = torch.tril_indices(row=2, col=2, offset=0)
                y_train_mat = torch.zeros((y_train.shape[0],2,2)).to(device)
                y_train[:,[1,2]] = y_train[:,[2,1]]

                y_train_mat[:,tril_indices[0],tril_indices[1]] = y_train[:,:]
                y_train_mat[:,tril_indices[1],tril_indices[0]] = y_train[:,:]

                r = torch.zeros_like(y_train_mat).to(device)
                r[:,0,0] = torch.cos(theta_p)
                r[:,0,1] = torch.sin(theta_p)
                r[:,1,0] = -torch.sin(theta_p)
                r[:,1,1] = torch.cos(theta_p)

                y_princ_mat =  r @ y_train_mat @ torch.transpose(r,1,2)

                #y_princ_mat = torch.from_numpy(rotate_tensor(y_train_mat.numpy(),theta_p.reshape(-1)))
                y_princ = y_princ_mat[:,[0,1],[0,1]].reshape([-1,2])
                            
                y_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
                y_scaler.fit(y_princ.cpu())
                y_princ = torch.from_numpy(y_scaler.scale_).to(device)*y_princ
                s = torch.from_numpy(y_scaler.scale_).to(device)*pred
                
                # rot_mat = torch.zeros_like(s_princ_mat)
                # rot_mat[:,:,0,0] = torch.cos(angles[:,:])
                # rot_mat[:,:,0,1] = torch.sin(angles[:,:])
                # rot_mat[:,:,1,0] = -torch.sin(angles[:,:])
                # rot_mat[:,:,1,1] = torch.cos(angles[:,:])

                # s = torch.transpose(rot_mat,2,3) @ s_princ_mat @ rot_mat
                # s_vec = s[:,:,[0,1,1],[0,1,0]].reshape([-1,3])
                

                # l = torch.reshape(pred,[t_pts,n_elems,6])

                # # Cholesky decomposition
                # L = torch.zeros([t_pts, n_elems, 3, 3])
                # tril_indices = torch.tril_indices(row=3, col=3, offset=0)
                # L[:, :, tril_indices[0], tril_indices[1]] = l[:,:]
                # # Tangent matrix
                # H = L @torch.transpose(L,2,3)

                # # Stress increment
                # d_s = (H @ d_e).squeeze()
                # s = torch.cumsum(d_s,0).reshape([-1,3])

                # s_ = copy.deepcopy(s.detach())
                # # Equivalent stress
                # mises = torch.sqrt(torch.square(s_[:,0]) - s_[:,0]*s_[:,1] + torch.square(s_[:,1]) + 3 * torch.square(s_[:,-1]))
                # # Yield transition
                # yield_transition = torch.sigmoid((torch.square(mises)-160**2))
                
                # yield_pt = torch.nonzero(yield_transition)

                # if yield_pt.shape[0] != 0:
                #     yield_pt = yield_pt[0]
                
                #     d_e_p = torch.zeros_like(s)
                #     d_e_p[yield_pt:,:] = d_e.squeeze().reshape([-1,3])[yield_pt:,:]
                
                #     # Plastic power
                #     w_p = torch.sum(s * d_e_p,-1)
                
                # else:
                #     w_p = torch.tensor(0.0)

                # l_wp = torch.mean(torch.nn.functional.relu(-w_p))

                
                # Stress at t=0
                s_0 = s[:9,:]
                l_0 = torch.mean(torch.square(s_0))
                
                # l_1 = torch.mean(torch.nn.functional.relu(-(d_e.transpose(2,3) @ H @ d_e)))

                # l_cholesky = torch.mean(nn.functional.relu(-torch.diagonal(L, offset=0, dim1=2, dim2=3)))

                # e_p = copy.deepcopy(e)
                # e_p[:,:,0:2] -= ((1/3)*torch.sum(e[:,:,0:2],-1,keepdim=True))
                # d_ep_dt = torch.cat([torch.zeros(1,n_elems,3),torch.diff(e_p,dim=0)/0.02])
                
                # w_ep = torch.sum(torch.reshape(pred,[t_pts,n_elems,3]) * d_ep_dt,-1)

                # l_w = torch.mean(torch.nn.functional.relu(-w_ep))
                # v,m = torch.std_mean(torch.from_numpy(dataloader.y.values),0)
                # w_x = 1/v[0]
                # w_y = 1/v[1]
                # w_xy = 1/v[2]

                # y_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
                # y_scaler.fit(y_train)
                # y_train = torch.from_numpy(y_scaler.scale_)*y_train
                # s_vec = torch.from_numpy(y_scaler.scale_)*s_vec
                

                # # Computing loss
                #loss = loss_fn(s_vec[:,0], y_train[:,0]) + loss_fn(s_vec[:,1], y_train[:,1]) + loss_fn(s_vec[:,2], y_train[:,2]) + l_0
                # loss = loss_fn(pred_princ[:,0],y_princ[:,0]) + loss_fn(pred_princ[:,1],y_princ[:,1]) + loss_fn(pred_eigen[:,0],y_eigen_vec[:,0]) + loss_fn(pred_eigen[:,1],y_eigen_vec[:,1]) + loss_fn(pred_eigen[:,2],y_eigen_vec[:,2]) + loss_fn(pred_eigen[:,3],y_eigen_vec[:,3]) + l_0
                
                loss = loss_fn(s[:,0],y_princ[:,0]) + loss_fn(s[:,1],y_princ[:,1])

            # loss = (loss_fn(s[:,0], y_train[:,0]) + loss_fn(s[:,1], y_train[:,1]) + loss_fn(s[:,2], y_train[:,2])) + l_1 + 500*l_cholesky
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()    
            # Backpropagation and weight's update
            optimizer.zero_grad(set_to_none=True)
            # loss.backward()
            # # Gradient clipping - as in https://github.com/pseeth/autoclip
            # g_norm.append(get_grad_norm(model))
            # clip_value = np.percentile(g_norm, 25)
            #nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

            #optimizer.step()
            #scheduler.step()
            #model.apply(constraints)

            # Saving loss values
            losses[batch] = loss.detach().item()
            l0_loss[batch] = 0
            l1_loss[batch] = 0
            ch_loss[batch] = 0
            wp_loss[batch] = 0
            I1_loss[batch] = 0
            

            g_norm.append(get_grad_norm(model))
        

            print('\r>Train: %d/%d' % (batch + 1, num_batches), end='')
              
        #-----------------------------
        return losses, np.mean(g_norm), l0_loss, l1_loss, ch_loss, wp_loss, I1_loss

    def test_loop(dataloader, model, loss_fn):

        #global w_virt

        num_batches = len(dataloader)
        test_losses = torch.zeros(num_batches)

        #n_vfs = v_strain.shape[0]
        # t_pts = dataloader.t_pts
        # n_elems = batch_size // t_pts
        model.eval()
        with torch.no_grad():

            for batch in range(num_batches):

                # Extracting variables for testing
                X_test, y_test, _, _, _, inc = dataloader[batch]

                t_pts = dataloader.t_pts[dataloader.tags[batch]]
                n_elems = dataloader.batch_sizes[dataloader.tags[batch]] // t_pts

                # Converting to pytorch tensors
                if dataloader.scaler_x is not None:
                    X_test = torch.from_numpy(dataloader.scaler_x.transform(X_test)).to(device)

                else:
                    X_test = torch.tensor(X_test, dtype=torch.float64).to(device)

                #X_test = X_test[~torch.any(X_test.isnan(),dim=1)]
                y_test = torch.from_numpy(y_test).to(device)

                theta_p = torch.from_numpy(inc[['theta_p']].values).reshape(-1).to(device)
                # theta_p = theta_p[~torch.any(theta_p.isnan(),dim=1)]
                # theta_p = torch.cat((torch.zeros(n_elems,1),theta_p),0)
                
                t = torch.from_numpy(inc['t'].values).reshape(-1,1).to(device)
                #dt = torch.diff(t,dim=0).reshape(-1,1)

                pred = model(X_test) # stress rate
                # pred = torch.cat((torch.zeros(n_elems,pred.shape[-1]),pred),0)
                # pred = torch.cumulative_trapezoid(pred.reshape(t_pts,n_elems,pred.shape[-1]),t.reshape(t_pts,n_elems,t.shape[-1]),dim=0).reshape(-1,pred.shape[-1])     # stress
                pred *= t.repeat(1,pred.shape[-1])
                # pred = torch.cumsum(pred.reshape(t_pts-1,n_elems,pred.shape[-1]),0).reshape(-1,pred.shape[-1])
        
                tril_indices = torch.tril_indices(row=2, col=2, offset=0)
                y_train_mat = torch.zeros((y_test.shape[0],2,2)).to(device)
                y_test[:,[1,2]] = y_test[:,[2,1]]

                y_train_mat[:,tril_indices[0],tril_indices[1]] = y_test[:,:]
                y_train_mat[:,tril_indices[1],tril_indices[0]] = y_test[:,:]

                r = torch.zeros_like(y_train_mat).to(device)
                r[:,0,0] = torch.cos(theta_p)
                r[:,0,1] = torch.sin(theta_p)
                r[:,1,0] = -torch.sin(theta_p)
                r[:,1,1] = torch.cos(theta_p)

                y_princ_mat =  r @ y_train_mat @ torch.transpose(r,1,2)

                #y_princ_mat = torch.from_numpy(rotate_tensor(y_train_mat.numpy(),theta_p.reshape(-1)))
                y_princ = y_princ_mat[:,[0,1],[0,1]].reshape([-1,2])
                
                y_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
                y_scaler.fit(y_princ.cpu())
                y_princ = torch.from_numpy(y_scaler.scale_).to(device)*y_princ
                s = torch.from_numpy(y_scaler.scale_).to(device)*pred
        
                # s_princ = torch.reshape(pred,[t_pts,n_elems,2])
                # s_princ_mat = torch.zeros([t_pts,n_elems,2,2])
                
                # s_princ_mat[:,:,0,0] = s_princ[:,:,0]
                # s_princ_mat[:,:,1,1] = s_princ[:,:,1]

                
                # rot_mat = torch.zeros_like(s_princ_mat)
                # rot_mat[:,:,0,0] = torch.cos(angles[:,:])
                # rot_mat[:,:,0,1] = torch.sin(angles[:,:])
                # rot_mat[:,:,1,0] = -torch.sin(angles[:,:])
                # rot_mat[:,:,1,1] = torch.cos(angles[:,:])

                # s = torch.transpose(rot_mat,2,3) @ s_princ_mat @ rot_mat
                # s_vec = s[:,:,[0,1,1],[0,1,0]].reshape([-1,3])
                
                # l = torch.reshape(pred,[t_pts,n_elems,6])

                # L = torch.zeros([t_pts, n_elems, 3, 3])
                # tril_indices = torch.tril_indices(row=3, col=3, offset=0)
                # L[:, :, tril_indices[0], tril_indices[1]] = l[:,:]
                # H = L @torch.transpose(L,2,3)

                # d_s = (H @ d_e).squeeze()
                # s = torch.cumsum(d_s,0).reshape([-1,3])

                # v,m = torch.std_mean(torch.from_numpy(train_generator.y.values),0)
                # w_x = 1/v[0]
                # w_y = 1/v[1]
                # w_xy = 1/v[2]

                # y_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
                # y_scaler.fit(y_test)
                # y_test = torch.from_numpy(y_scaler.scale_)*y_test
                # s_vec = torch.from_numpy(y_scaler.scale_)*s_vec
               
                # Computing losses
                #test_loss = loss_fn(s_vec[:,0], y_test[:,0]) + loss_fn(s_vec[:,1], y_test[:,1]) + loss_fn(s_vec[:,2], y_test[:,2])
                #test_loss = loss_fn(pred_princ[:,0],y_princ[:,0]) + loss_fn(pred_princ[:,1],y_princ[:,1]) + loss_fn(pred_eigen[:,0],y_eigen_vec[:,0]) + loss_fn(pred_eigen[:,1],y_eigen_vec[:,1]) + loss_fn(pred_eigen[:,2],y_eigen_vec[:,2]) + loss_fn(pred_eigen[:,3],y_eigen_vec[:,3])
               
                test_loss = loss_fn(s[:,0],y_princ[:,0]) + loss_fn(s[:,1],y_princ[:,1])
                
                test_losses[batch] = test_loss
               

                print('\r>Test: %d/%d' % (batch + 1, num_batches), end='')

        #get_sbvfs(copy.deepcopy(model), dataloader, isTrain=False)
        return test_losses
#----------------------------------------------------

    # Default floating point precision for pytorch
    torch.set_default_dtype(torch.float64)

    if torch.cuda.is_available():  
        dev = "cuda:0" 
    else:  
        dev = "cpu"  
    device = torch.device(dev) 

    # Specifying random seed
    os.environ['PYTHONHASHSEED']=str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Loading data
    df_list, _ = load_dataframes(TRAIN_MULTI_DIR)

    # Merging training data
    data = pd.concat(df_list, axis=0, ignore_index=True)    

    #T_PTS = len(set(data['t']))
    
    # if T_PTS==DATA_SAMPLES:
    # Reorganizing dataset by tag, subsequent grouping by time increment
    data_by_tag = [df for _, df in data.groupby(['tag'])]
    random.shuffle(data_by_tag)
    data_by_t = [[df for _, df in group.groupby(['t'])] for group in data_by_tag]
    #random.shuffle(data_by_t)
    data_by_batches = list(itertools.chain(*data_by_t))
    data_by_batches = [df.sort_values('id') for df in data_by_batches]
    #random.shuffle(data_by_batches)
     
    trials = [list(set(df['tag']))[0] for df in data_by_tag]
    
    time_points = dict.fromkeys(trials)
    
    for i, (k, _) in enumerate(time_points.items()):
        if k == data_by_t[i][0]['tag'].values[0]:
            time_points[k] = len(data_by_t[i])

    batch_sizes = dict.fromkeys(trials)

    for i, (k, _) in enumerate(batch_sizes.items()):
        if k == data_by_t[i][0]['tag'].values[0]:
            batch_sizes[k] = len(data_by_t[i][0]) * time_points[k]
    
    # batch_size = len(data_by_batches[0]) * T_PTS

    data = pd.concat(data_by_batches).reset_index(drop=True)

    #trials = list(set(data['tag'].values))
    random.shuffle(trials)
    test_trials = random.sample(trials, math.ceil(len(trials)*TEST_SIZE))
    train_trials = list(set(trials).difference(test_trials))

    # Performing test/train split
    partition = {"train": None, "test": None}

    partition['train'] = data[data['tag'].isin(train_trials)].index.tolist()
    partition['test'] = data[data['tag'].isin(test_trials)].index.tolist()

    # Selecting model features
    X, y, f, coord, info = select_features_multi(data)
    
    del data
    del df_list
    del data_by_batches
    del data_by_t
    del data_by_tag
    gc.collect()

    # Preparing data generators for mini-batch training

    #partition['train'] = data[data['tag']=='m80_b80_x'].index.tolist()
    train_batches = {k: batch_sizes[k] for k in train_trials}
    train_tpts = {k: time_points[k] for k in train_trials}
    train_generator = DataGenerator(X, y, f, coord, info, partition["train"], train_batches, shuffle=False, std=True, t_pts=train_tpts)
    
    test_batches = {k: batch_sizes[k] for k in test_trials}
    test_tpts = {k: time_points[k] for k in test_trials}
    test_generator = DataGenerator(X, y, f, coord, info, partition['test'], test_batches, shuffle=False, std=False, t_pts=test_tpts, scaler=train_generator.scaler_x)

    # Model variables
    N_INPUTS = X.shape[1]
    N_OUTPUTS = 2

    N_UNITS = [25,20,15,10]
    H_LAYERS = len(N_UNITS)

    WANDB_LOG = True

    model_1 = NeuralNetwork(N_INPUTS, N_OUTPUTS, N_UNITS, H_LAYERS)

    model_1.apply(init_weights)
    model_1.to(device)

    # clip_value = 100
    # for p in model_1.parameters():
    #     p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))

    # Training variables
    epochs = 5000
    # Optimization variables
    # Optimization variables
    learning_rate = 0.02
    lr_mult = 1.0

    params = layer_wise_lr(model_1, lr_mult=lr_mult, learning_rate=learning_rate)

    loss_fn = torch.nn.MSELoss()

    
    weight_decay=0.0

    optimizer = torch.optim.AdamW(params=params, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, cooldown=5, factor=0.88, min_lr=[params[i]['lr']*0.00005 for i in range(len(params))])
    #[params[i]['lr']*0.025 for i in range(len(params))]
    #scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,base_lr=0.002, max_lr=0.007, mode='exp_range',gamma=0.99994,cycle_momentum=False,step_size_up=len(train_generator)*20,step_size_down=len(train_generator)*200)
    #constraints=weightConstraint(cond='plastic')

    # Creates a GradScaler once at the beginning of training.
    USE_AMP = True
    scaler = torch.cuda.amp.GradScaler()

    # Container variables for history purposes
    train_loss = []
    v_work = []
    val_loss = []
    epochs_ = []

    l0_loss = []
    l1_loss = []
    ch_loss = []
    wp_loss = []
    I1_loss = []
    # j2_loss = []

    # Initializing the early_stopping object
    early_stopping = EarlyStopping(patience=500, path='temp/checkpoint.pt', verbose=True)

    if WANDB_LOG:
        config = {
            "inputs": N_INPUTS,
            "outputs": N_OUTPUTS,
            "hidden_layers": H_LAYERS,
            "hidden_units": "/".join(str(x) for x in N_UNITS),        
            "epochs": epochs,
            "lr": learning_rate,
            "l2_reg": weight_decay
        }
        run=wandb.init(project="direct_training_principal_inc_crux", entity="rmbl",config=config)
        wandb.watch(model_1,log='all')

    for t in range(epochs):

        print('\r--------------------\nEpoch [%d/%d]' % (t + 1, epochs))

        epochs_.append(t+1)

        #Shuffling batches
        for generator in [train_generator,test_generator]:
            generator.on_epoch_end()

        # Train loop
        start_train = time.time()

        #--------------------------------------------------------------
        batch_losses, grad_norm, l_0, l_1, l_ch, l_wp, l_I1 = train_loop(train_generator, model_1, loss_fn, optimizer)
        #--------------------------------------------------------------

        train_loss.append(torch.mean(batch_losses))
        l0_loss.append(torch.mean(l_0))
        l1_loss.append(torch.mean(l_1))
        ch_loss.append(torch.mean(l_ch))
        wp_loss.append(torch.mean(l_wp))
        I1_loss.append(torch.mean(l_I1))
        
        
        end_train = time.time()

        #Apply learning rate scheduling if defined
        try:
            scheduler.step(train_loss[t])
            print('. t_loss: %.6e -> lr: %.4e | l0: %.4e | l1: %.4f | lw: %.4f -- %.3fs' % (train_loss[t], scheduler._last_lr[0], l0_loss[t], l1_loss[t], ch_loss[t], end_train - start_train))
        except:
            print('. t_loss: %.6e -> | l0: %.4e | l1: %.4f | lw: %.4f -- %.3fs' % (train_loss[t], l0_loss[t], l1_loss[t], ch_loss[t],end_train - start_train))

        # Test loop
        start_test = time.time()

        #-----------------------------------------------------------------------------------
        batch_val_losses = test_loop(test_generator, model_1, loss_fn)
        #-----------------------------------------------------------------------------------

        val_loss.append(torch.mean(batch_val_losses).item())

        end_test = time.time()

        print('. v_loss: %.6e -- %.3fs' % (val_loss[t], end_test - start_test))

        if t > 200:
            early_stopping(val_loss[t], model_1)

        if WANDB_LOG:
            wandb.log({
                'epoch': t,
                'l_rate': scheduler._last_lr[0],
                'train_loss': train_loss[t],
                'test_loss': val_loss[t],
                's(0)_error': l0_loss[t],
                'drucker_error': l1_loss[t],
                'cholesky_error': ch_loss[t],
                'plastic_power': wp_loss[t],
                'I1_error': I1_loss[t],
                #'j2_error': j2_loss[t]
            })


        if  early_stopping.early_stop:
            print("Early stopping")
            break
    
    print("Done!")

    # load the last checkpoint with the best model
    #model_1.load_state_dict(torch.load('temp/checkpoint.pt'))

    epochs_ = np.reshape(np.array(epochs_), (len(epochs_),1))
    train_loss = np.reshape(np.array(train_loss), (len(train_loss),1))
    val_loss = np.reshape(np.array(val_loss), (len(val_loss),1))

    history = pd.DataFrame(np.concatenate([epochs_, train_loss, val_loss], axis=1), columns=['epoch','loss','val_loss'])


    task = r'%s-[%i-%ix%i-%i]-%s-%i-VFs' % (run.name,N_INPUTS, N_UNITS[0], H_LAYERS, N_OUTPUTS, TRAIN_MULTI_DIR.split('/')[-2], count_parameters(model_1))

    output_task = 'outputs/' + TRAIN_MULTI_DIR.split('/')[-2] + '_sbvf_abs_direct'
    output_loss = 'outputs/' + TRAIN_MULTI_DIR.split('/')[-2] + '_sbvf_abs_direct/loss/'
    output_stats = 'outputs/' + TRAIN_MULTI_DIR.split('/')[-2] + '_sbvf_abs_direct/stats/'
    output_models = 'outputs/' + TRAIN_MULTI_DIR.split('/')[-2] + '_sbvf_abs_direct/models/'
    output_val = 'outputs/' + TRAIN_MULTI_DIR.split('/')[-2] + '_sbvf_abs_direct/val/'
    output_logs = 'outputs/' + TRAIN_MULTI_DIR.split('/')[-2] + '_sbvf_abs_direct/logs/'

    directories = [output_task, output_loss, output_stats, output_models, output_val, output_logs]

    for dir in directories:
        try:
            os.makedirs(dir)

        except FileExistsError:
            pass

    history.to_csv(output_loss + task + '.csv', sep=',', encoding='utf-8', header='true')

    plot_history(history, output_loss, True, task)

    torch.save(model_1.state_dict(), output_models + task + '.pt')
    
    #joblib.dump([VFs, W_virt], 'sbvfs.pkl')
    if train_generator.std == True:
        joblib.dump(train_generator.scaler_x, output_models + task + '-scaler_x.pkl')

    if WANDB_LOG:
        # 3️⃣ At the end of training, save the model artifact
        # Name this artifact after the current run
        task_ = r'__%i-%ix%i-%i__%s-direct' % (N_INPUTS, N_UNITS[0], H_LAYERS, N_OUTPUTS, TRAIN_MULTI_DIR.split('/')[-2])
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

    train()

    # Deleting temp folder
    try:
        shutil.rmtree('temp/')
    except FileNotFoundError:
        pass

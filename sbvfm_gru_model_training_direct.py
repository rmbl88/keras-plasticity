# ---------------------------------
#    Library and function imports
# ---------------------------------
import cProfile, pstats
from glob import glob
from msvcrt import kbhit
import os
import shutil
import joblib
from pytools import F
from constants import *
from functions import (
    CoVWeightingLoss,
    GRUModel,
    global_dof,
    layer_wise_lr,
    standardize_data,
    plot_history
    )
from functions import (
    weightConstraint,
    EarlyStopping,
    NeuralNetwork
    )

import tensorflow as tf
import pandas as pd
import random
import numpy as np
import math
import torch
import time
import wandb
from torch.autograd.functional import jacobian
import time
from sklearn import preprocessing
import time
import random
from tkinter import *
import matplotlib.pyplot as plt
import gc
import pyarrow.parquet as pq
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
import pytorch_warmup as warmup
from warmup_scheduler import GradualWarmupScheduler
import math
import glob
# ----------------------------------------
#        Class definitions
# ----------------------------------------
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class CruciformDataset(torch.utils.data.Dataset):
    def __init__(self, trials, root_dir, data_dir, features, outputs, info, scaler_x=None, transform=None, seq_len=1):
        self.seq_len = seq_len
        self.trials = trials
        self.root_dir = root_dir
        self.data_dir = data_dir
        self.scaler_x = scaler_x
        self.features = features
        self.outputs = outputs
        self.info = info
        self.transform = transform
        self.files = [os.path.join(self.root_dir, self.data_dir , trial + '.parquet') for trial in self.trials]
        self.data = [pq.ParquetDataset(file).read_pandas(columns=self.features+self.outputs+['tag','t','id','s1','s2','fxx_t','fyy_t','area','sxx_t','syy_t']).to_pandas() for file in self.files]

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, idx):
        
        x = torch.from_numpy(self.data[idx][self.features].dropna().values).float()
        y = torch.from_numpy(self.data[idx][self.outputs].dropna().values).float()
        t = torch.from_numpy(self.data[idx]['t'].values).float()
        #e = torch.from_numpy(self.data[idx][['dep_1','dep_2']].dropna().values).float()
        # f = torch.from_numpy(self.data[idx][['fxx_t','fyy_t']].values).float()
        # a = torch.from_numpy(self.data[idx]['area'].values).reshape(-1,1).float()

        t_pts = len(list(set(t.numpy())))
        n_elems = len(set(self.data[idx]['id'].values))

        # Adding a padding of zeros to the input data in order to make predictions start at zero
        #pad_zeros = torch.zeros(self.seq_len * n_elems, x.shape[-1])
        
        pad_zeros = torch.zeros((self.seq_len-1) * n_elems, x.shape[-1])
        x = torch.cat([pad_zeros, x], 0)

        if self.transform != None:
            
            #dt = torch.diff(t.reshape(t_pts,n_elems,-1),0)
            x = self.transform(x)
            
        #x = self.rolling_window(x.reshape(t_pts + self.seq_len,n_elems,-1), seq_size=self.seq_len)[:,:-1]
        x = self.rolling_window(x.reshape(t_pts + self.seq_len-1, n_elems,-1), seq_size=self.seq_len)
        x = x.reshape(-1,*x.shape[2:])
        #t = self.rolling_window(t.reshape(t_pts,n_elems,-1), seq_size=self.seq_len)

        #y = y.reshape(t_pts-1,n_elems,-1)[self.seq_len-1:].reshape(-1,y.shape[-1])
        #y = y.reshape(t_pts,n_elems,-1)[self.seq_len-1:].permute(1,0,2)
        y = y.reshape(t_pts,n_elems,-1).permute(1,0,2)
        y = y.reshape(-1,y.shape[-1])

        idx_ = torch.randperm(x.shape[0])

        return x[idx_], y[idx_], t, t_pts, n_elems      
    
    def rolling_window(self, x, seq_size, step_size=1):
    # unfold dimension to make our rolling window
        return x.unfold(0,seq_size,step_size).permute(1,0,3,2)
        #return x.unfold(0,seq_size,step_size).permute(0,1,3,2).reshape(-1,seq_size,x.shape[-1])

class MinMaxScaler(object):
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def __call__(self, x):

        x_std = (x - self.min) / (self.max - self.min)
        x_scaled = x_std * (self.max - self.min) + self.min

        return x_scaled

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = torch.as_tensor(mean)
        self.std = torch.as_tensor(std)

    def __call__(self, x):

        x_std = (x - self.mean) / self.std

        return x_std
        
# -------------------------------
#       Method definitions
# -------------------------------

def batch_jacobian(f, x):
    f_sum = lambda x: torch.sum(f(x[:,-1]), axis=0)
    return jacobian(f_sum, x[:,-1],create_graph=True).permute(1,0,2)

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
        if isinstance(m, torch.nn.Linear) and (m.bias != None):
            torch.nn.init.kaiming_uniform_(m.weight) #RELU
            #torch.nn.init.xavier_normal(m.weight)
            #torch.nn.init.zeros_(m.bias)
            #torch.nn.init.ones_(m.bias)
            m.bias.data.fill_(0.01)

    def train_loop(dataloader, model, l_fn, optimizer, epoch):
        '''
        Custom loop for neural network training, using mini-batches

        '''

        data_iter = iter(dataloader)

        # next_batch = data_iter.__next__()  # start loading the first batch
        # next_batch = [_.cuda(non_blocking=True) for _ in next_batch]

        num_batches = len(dataloader)
        
        losses = []
        f_loss = []
        triax_loss = []
        l_loss = []
               
        model.train()
        #l_fn.to_train()
        
        for batch_idx in range(len(dataloader)):
    
            # batch = next_batch

            # if batch_idx + 1 != len(dataloader): 
            #     # start copying data of next batch
            #     next_batch = data_iter.__next__()
            #     next_batch = [_.cuda(non_blocking=True) for _ in next_batch]

            # Extracting variables for training
            X_train, y_train, _, t_pts, n_elems = data_iter.__next__()
            
            X_train = X_train.squeeze(0)
            y_train = y_train.squeeze(0)

            x_batches = X_train.split(X_train.shape[0]//32)
            y_batches = y_train.split(y_train.shape[0]//32)
            #t = t.squeeze(0).to(device)
            #e = e.squeeze(0).to(device)
            #f = f.squeeze(0).to(device)
            #a = a.squeeze(0).to(device)

            for i, batch in enumerate(x_batches):
                model.init_hidden(batch.size(0),device)
                #batch.requires_grad_(True)
                with torch.autocast(device_type='cuda', dtype=torch.float32, enabled=USE_AMP):
                    pred, _ = model(batch.to(device)) # stress rate
                
                

                # J = torch.zeros(pred.shape[0],3,3)
                # for i in range(3):
                #     v = torch.ones_like(pred).cuda()
                #     v[:,i] *= 0.
                #     pred.backward(v, retain_graph=True)
                #     J[:,:,i] = batch.grad[:,-1]
                #     batch.grad.zero_()
                

                #jac = torch.stack([torch.autograd.grad(pred[:, j].sum(), batch, create_graph=True)[0][:,-1] for j in range(3)],-1)
                # J = torch.zeros(pred.shape[0],3,3).cuda()
                # for j in range(3):
                #     output = torch.zeros(pred.shape[0],3).cuda()
                #     output[:,j] = 1.
                #     J[:,:,j:j+1] = torch.autograd.grad(pred, batch, grad_outputs=output, create_graph=True)[0][:,-1].unsqueeze(-1)

            #dt = torch.diff(t.reshape(t_pts,n_elems,1),dim=0)
            #s_princ_ = torch.zeros([t_pts,n_elems,2]).to(device)
            #s_princ_[1:,:,:] = pred.reshape(t_pts-1,n_elems,pred.shape[-1])*dt
            #s_princ_ = torch.cumsum(s_princ_,0)

            #f_ = (torch.sum(s_princ_*a.reshape([t_pts,n_elems,1]),1)/30.0).unsqueeze(1).repeat(1,n_elems,1).reshape(-1,s_princ_.shape[-1])

            
            #s_princ_ord = torch.sort(s_princ_,-1,descending=True).values
            #triax = (math.sqrt(2)/3)*(s_princ_ord[:,:,0]+s_princ_ord[:,:,1])/(s_princ_ord[:,:,0]-s_princ_ord[:,:,1]+1e-12)
            #l_triax = torch.nn.functional.relu(torch.abs(triax.reshape(-1,1))-2/3)
            #de_p = e-((1/3)*torch.sum(e,1)).reshape(-1,1).repeat(1,e.shape[-1])
            #wm = torch.sum(s_princ_*de_p,1).reshape(-1)


            # y_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
            # y_scaler.fit(y_train.cpu())
            # dot_y_princ = torch.from_numpy(y_scaler.scale_).to(device)*y_train
            # s = torch.from_numpy(y_scaler.scale_).to(device)*pred
                
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
                
                with torch.autocast(device_type='cuda', dtype=torch.float32, enabled=USE_AMP):
                    #l_triaxiality = torch.mean(torch.nn.functional.relu(torch.abs(triaxiality)-2/3)
                    #l_tuples = [(pred[:,0], y_batches[i][:,0].to(device)), (pred[:,1], y_batches[i][:,1].to(device)), (pred[:,1], y_batches[i][:,1].to(device))]
                    
                    #l_jac = torch.sum(torch.square(torch.cat([jac[:,2,[0,1]],jac[:,[0,1],2]],1)))
                    loss = l_fn(pred, y_batches[i].to(device))
                    #loss = l_fn(l_tuples)
                
                scaler.scale(loss/ITERS_TO_ACCUMULATE).backward()

                # scaler.unscale_(optimizer)

                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
            
                if ((i + 1) % ITERS_TO_ACCUMULATE == 0) or (i + 1 == len(x_batches)):    
                    
                    scaler.step(optimizer)
                    scaler.update()
                    
                    optimizer.zero_grad(set_to_none=True)

                    warmup_lr.step()
            
                # Saving loss values
                losses.append(loss.item())
                f_loss.append(0.0)
                triax_loss.append(0.0)
                l_loss.append(0.0  )

            print('\r>Train: %d/%d' % (batch_idx + 1, num_batches), end='')
              
        #-----------------------------
        return losses, f_loss, l_loss, triax_loss

    def test_loop(dataloader, model, l_fn):

        data_iter = iter(dataloader)

        # next_batch = data_iter.__next__()  # start loading the first batch
        # next_batch = [_.cuda(non_blocking=True) for _ in next_batch]

        num_batches = len(dataloader)
        test_losses = []

        model.eval()
        #l_fn.to_eval()

        with torch.no_grad():

            for batch_idx in range(len(dataloader)):

                # if batch_idx + 1 != len(dataloader): 
                #     # start copying data of next batch
                #     next_batch = data_iter.__next__()
                #     next_batch = [_.cuda(non_blocking=True) for _ in next_batch]
            
                X_test, y_test, _, t_pts, n_elems = data_iter.__next__()
            
                X_test = X_test.squeeze(0).to(device)
                y_test = y_test.squeeze(0).to(device)
                # t = t.squeeze(0).to(device)
                # e = e.squeeze(0).to(device)
                # f = f.squeeze(0).to(device)
                # a = a.squeeze(0).to(device)

                x_batches = X_test.split(X_test.shape[0]//32)
                y_batches = y_test.split(y_test.shape[0]//32)

                for i, batch in enumerate(x_batches):
                    model.init_hidden(batch.size(0),device)
                    with torch.autocast(device_type='cuda', dtype=torch.float32, enabled=USE_AMP):
                        pred, _ = model(batch) # stress rate
                    
                # dt = torch.diff(t.reshape(t_pts,n_elems,1),dim=0)
                # s_princ_ = torch.zeros([t_pts,n_elems,2]).to(device)
                # s_princ_[1:,:,:] = pred.reshape(t_pts-1,n_elems,pred.shape[-1])*dt
                # s_princ_ = torch.cumsum(s_princ_,0)

                # f_ = (torch.sum(s_princ_*a.reshape([t_pts,n_elems,1]),1)/30.0).unsqueeze(1).repeat(1,n_elems,1).reshape(-1,s_princ_.shape[-1])
                # s_princ_ord = torch.sort(s_princ_,-1,descending=True).values
                # triax = (math.sqrt(2)/3)*(s_princ_ord[:,:,0]+s_princ_ord[:,:,1])/(s_princ_ord[:,:,0]-s_princ_ord[:,:,1]+1e-12)
                # l_triax = torch.nn.functional.relu(torch.abs(triax.reshape(-1,1))-2/3)
                
                
                # y_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
                # y_scaler.fit(y_test.cpu().detach().numpy())
                # dot_y_princ = torch.from_numpy(y_scaler.scale_).to(device)*y_test
                # s = torch.from_numpy(y_scaler.scale_).to(device)*pred
        
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
                
                    with torch.autocast(device_type='cuda', dtype=torch.float32, enabled=USE_AMP):
                        
                        #l_tuples = [(pred[:,0], y_test[:,0]), (pred[:,1], y_test[:,1]), (l_triax,)]
                    
                        #test_loss = l_fn(l_tuples)
                        # test_loss = l_fn(pred[:,0], y_test[:,0]) + l_fn(pred[:,1], y_test[:,1])
                        test_loss = l_fn(pred, y_batches[i].to(device))
                
                    test_losses.append(test_loss.item())
                    
                    # del X_test, y_test
                    # gc.collect()

                print('\r>Test: %d/%d' % (batch_idx + 1, num_batches), end='')

        return test_losses
#----------------------------------------------------
    # Disabling Debug APIs
    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)
    torch.autograd.profiler.emit_nvtx(False)

    # Default floating point precision for pytorch
    torch.set_default_dtype(torch.float32)

    if torch.cuda.is_available():  
        dev = "cuda:0"
        KWARGS = {'num_workers': 0, 'pin_memory': False} 
    else:  
        dev = "cpu"  
        KWARGS = {'num_workers': 0}
    
    device = torch.device(dev)

    torch.cuda.empty_cache() 

    # Specifying random seed
    os.environ['PYTHONHASHSEED']=str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    trials = pd.read_csv(os.path.join(TRAIN_MULTI_DIR,'t_trials.csv'), index_col=False, header=0)
    trials = list(trials['0'])
    trials = random.sample(trials,len(trials))

    test_trials = random.sample(trials, math.ceil(len(trials)*TEST_SIZE))
    train_trials = list(set(trials).difference(test_trials))

    # Defining variables of interest

    FEATURES = ['exx_t','eyy_t','exy_t']
    OUTPUTS = ['sxx_t','syy_t','sxy_t']
    INFO = ['tag','inc','t','theta_ep','s1','s2','theta_sp','fxx_t','fyy_t']

    # Gathering statistics on training dataset

    #file_list = []
    df_list = []

    dir = os.path.join(TRAIN_MULTI_DIR,'processed/')
    # for r, d, f in os.walk(dir):
    #     for file in f:
    #         if '.csv' or '.parquet' in file:
    #             file_list.append(dir + file)

    file_list = glob.glob(os.path.join(dir, f'*.parquet'))

    df_list = [pq.ParquetDataset(file).read_pandas(columns=['tag']+FEATURES).to_pandas() for file in tqdm(file_list,desc='Importing dataset files',bar_format=FORMAT_PBAR)]

    raw_data = pd.concat(df_list)
    input_data = raw_data[raw_data['tag'].isin(train_trials)].drop('tag',1).dropna()

    min = torch.min(torch.from_numpy(input_data.values).float(),0).values
    max = torch.max(torch.from_numpy(input_data.values).float(),0).values
    std, mean = torch.std_mean(torch.from_numpy(input_data.values),0)

    # Cleaning workspace from useless variables
    del df_list
    del file_list
    del input_data
    gc.collect()

    # Defining data transforms - normalization and noise addition 
    transform = transforms.Compose([
        #MinMaxScaler(min,max),
        Normalize(mean.tolist(), std.tolist()),
        #transforms.RandomApply([AddGaussianNoise(0., 1.)],p=0.15)
    ])

    # Preparing dataloaders for mini-batch training
    SEQ_LEN = 4

    train_dataset = CruciformDataset(train_trials, TRAIN_MULTI_DIR, 'processed', FEATURES, OUTPUTS, INFO, transform=transform, seq_len=SEQ_LEN)
    test_dataset = CruciformDataset(test_trials, TRAIN_MULTI_DIR, 'processed', FEATURES, OUTPUTS, INFO, transform=transform, seq_len=SEQ_LEN)
    
    train_dataloader = DataLoader(train_dataset, shuffle=True,**KWARGS)
    test_dataloader = DataLoader(test_dataset, **KWARGS)
    
    # Model variables
    N_INPUTS = len(FEATURES)
    N_OUTPUTS = len(OUTPUTS)
    
    N_UNITS = [32]
    H_LAYERS = 2

    ITERS_TO_ACCUMULATE = 1

    # Automatic mixed precision
    USE_AMP = True

    # WANDB logging
    WANDB_LOG = True

    model_1 = GRUModel(input_dim=N_INPUTS, hidden_dim=N_UNITS, layer_dim=H_LAYERS, output_dim=N_OUTPUTS)

    model_1.apply(init_weights)
    model_1.to(device)

    # clip_value = 100
    # for p in model_1.parameters():
    #     p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))

    # Training variables
    epochs = 10000

    # Optimization variables
    learning_rate = 0.005
    lr_mult = 1.0

    #params = layer_wise_lr(model_1, lr_mult=lr_mult, learning_rate=learning_rate)
    
    #l_fn = CoVWeightingLoss(device=device, n_losses=3)
    l_fn = torch.nn.MSELoss()

    weight_decay = 0.001

    optimizer = torch.optim.AdamW(params=model_1.parameters(), weight_decay=weight_decay, lr=learning_rate)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=20, cooldown=8, factor=0.88, min_lr=[params[i]['lr']*0.00005 for i in range(len(params))])
       
    steps_warmup = 1920 # 5 epochs
    steps_annealing = (epochs - (steps_warmup // (len(train_dataloader)*32 // ITERS_TO_ACCUMULATE))) * (len(train_dataloader)*32 // ITERS_TO_ACCUMULATE)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps_annealing, eta_min=1e-3)
    warmup_lr = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=steps_warmup, after_scheduler=scheduler)
    
    #scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,base_lr=0.002, max_lr=0.007, mode='exp_range',gamma=0.99994,cycle_momentum=False,step_size_up=len(train_generator)*20,step_size_down=len(train_generator)*200)
    #constraints=weightConstraint(cond='plastic')

    # Creates a GradScaler once at the beginning of training.
    scaler = torch.cuda.amp.GradScaler()

    # Container variables for history purposes
    train_loss = []
    val_loss = []
    epochs_ = []

    f_loss = []
    l_loss = []
    triax_loss = []

    alpha1 = []
    alpha2 = []
    alpha3 = []
    alpha4 = []

    if WANDB_LOG:
        config = {
            "inputs": N_INPUTS,
            "outputs": N_OUTPUTS,
            "hidden_layers": H_LAYERS,
            "hidden_units": N_UNITS,        
            "epochs": epochs,
            "lr": learning_rate,
            "l2_reg": weight_decay
        }
        run=wandb.init(project="direct_training_principal_inc_crux_lstm", entity="rmbl",config=config)
        wandb.watch(model_1,log='all')
    
    # Initializing the early_stopping object
    
    if WANDB_LOG:
        path=f'temp/{run.name}/checkpoint.pt'
        try:
            os.makedirs(f'./temp/{run.name}')
        except FileExistsError:
            pass
    else:
        path=f'temp/checkpoint.pt'
        try:
            os.makedirs(f'./temp')
        except FileExistsError:
            pass
    

    early_stopping = EarlyStopping(patience=500, path=path, verbose=True)

    for t in range(epochs):

        print('\r--------------------\nEpoch [%d/%d]' % (t + 1, epochs))

        epochs_.append(t+1)

        # Train loop
        start_train = time.time()

        #--------------------------------------------------------------
        batch_losses, f_l, l_, t_l = train_loop(train_dataloader, model_1, l_fn, optimizer,t)
        #--------------------------------------------------------------

        train_loss.append(np.mean(batch_losses))
        f_loss.append(np.mean(f_l))
        l_loss.append(np.mean(l_))
        triax_loss.append(np.mean(t_l))
        
        alpha1.append(0.0)
        alpha2.append(0.0)
        alpha3.append(0.0)
        alpha4.append(0.0)

        end_train = time.time()

        #Apply learning rate scheduling if defined
        try:
            #scheduler.step(train_loss[t])
            print('. t_loss: %.6e -> lr: %.4e -- %.3fs \n\nl_mse: %.4e | f_l: %.4e | t_l: %.4e \na1: %.3e | a2: %.3e | a3: %.3e | a4: %.3e\n' % (train_loss[t], warmup_lr._last_lr[0], end_train - start_train, l_loss[t], f_loss[t], triax_loss[t], alpha1[t], alpha2[t], alpha3[t], alpha4[t]))
        except:
            print('. t_loss: %.6e -> | wm_l: %.4e -- %.3fs' % (train_loss[t], f_loss[t], end_train - start_train))

        # Test loop
        start_test = time.time()

        #-----------------------------------------------------------------------------------
        batch_val_losses = test_loop(test_dataloader, model_1, l_fn)
        #-----------------------------------------------------------------------------------

        val_loss.append(np.mean(batch_val_losses))

        end_test = time.time()

        print('. v_loss: %.6e -- %.3fs' % (val_loss[t], end_test - start_test))

        if t > 200:
            early_stopping(val_loss[t], model_1)

        if WANDB_LOG:
            wandb.log({
                'epoch': t,
                'l_rate': warmup_lr._last_lr[0],
                'train_loss': train_loss[t],
                'test_loss': val_loss[t],
                'l_jac': f_loss[t],
                't_loss': triax_loss[t],
                'mse_loss': l_loss[t],
                'alpha_1': alpha1[t],
                'alpha_2': alpha2[t],
                'alpha_3': alpha3[t],
                'alpha_4': alpha4[t]
            })

        if  early_stopping.early_stop:
            print("Early stopping")
            break
    
    print("Done!")

    # load the last checkpoint with the best model
    model_1.load_state_dict(torch.load(f'temp/{run.name}/checkpoint.pt'))

    epochs_ = np.reshape(np.array(epochs_), (len(epochs_),1))
    train_loss = np.reshape(np.array(train_loss), (len(train_loss),1))
    val_loss = np.reshape(np.array(val_loss), (len(val_loss),1))

    history = pd.DataFrame(np.concatenate([epochs_, train_loss, val_loss], axis=1), columns=['epoch','loss','val_loss'])

    task = f"{run.name}-[{N_INPUTS}-GRUx{H_LAYERS}-{*N_UNITS,}-{N_OUTPUTS}]-{TRAIN_MULTI_DIR.split('/')[-2]}-{count_parameters(model_1)}-VFs"
    #task = r'%s-[%i-%ix%i-%i]-%s-%i-VFs' % (run.name,N_INPUTS, N_UNITS[0], H_LAYERS, N_OUTPUTS, TRAIN_MULTI_DIR.split('/')[-2], count_parameters(model_1))

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

    #plot_history(history, output_loss, True, task)

    torch.save(model_1.state_dict(), output_models + task + '.pt')
    
    scaler_dict = {
        'type': 'standard',
        'stat_vars': [std, mean]
    }

    joblib.dump(scaler_dict, output_models + task + '-scaler_x.pkl')
    joblib.dump([FEATURES, OUTPUTS, INFO, N_UNITS, H_LAYERS, SEQ_LEN], output_models + run.name + '-arch.pkl')

    if WANDB_LOG:
        # 3️⃣ At the end of training, save the model artifact
        # Name this artifact after the current run
        task_ = f"{run.name}-[{N_INPUTS}-GRUx{H_LAYERS}-{*N_UNITS,}-{N_OUTPUTS}]-{TRAIN_MULTI_DIR.split('/')[-2]}-direct"
        #task_ = r'__%i-%ix%i-%i__%s-direct' % (N_INPUTS, N_UNITS[0], H_LAYERS, N_OUTPUTS, TRAIN_MULTI_DIR.split('/')[-2])
        model_artifact_name = run.id + '_' + run.name + task_
        # Create a new artifact, which is a sample dataset
        model = wandb.Artifact(model_artifact_name, type='model')
        # Add files to the artifact, in this case a simple text file
        model.add_file(local_path=output_models + task + '.pt')
        model.add_file(output_models + task + '-scaler_x.pkl')
        model.add_file(output_models + run.name + '-arch.pkl')
        # Log the model to W&B
        run.log_artifact(model)
        # Call finish if you're in a notebook, to mark the run as done
        run.finish()

        # Deleting temp folder
    try:
        shutil.rmtree(f'./temp/{run.name}')
    except FileNotFoundError:
        pass

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

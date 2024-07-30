# ---------------------------------
#    Library and function imports
# ---------------------------------
import os
import shutil
import sys
import joblib
import pandas as pd
from loss import(
    DataLoss, 
    instantiate_losses
)
from functions import (
    CosineAnnealingWarmupRestarts,
    GRUModel,
    EarlyStopping,
    GRUModelCholesky,
    GradNormLogger,
    get_data_stats,
    get_data_transform,
    train_test_split,
)

from io_funcs import (
    load_config,
    save_config
)

import random
import numpy as np
import torch
import time
import wandb
import time
import random

import pyarrow.parquet as pq
from torch.utils.data import DataLoader
from torchvision import transforms
from warmup_scheduler import GradualWarmupScheduler

from loss_aggregator import Relobralo
from autoclip.torch import QuantileClip
from dotenv import load_dotenv
from functorch import jacrev
import copy

from mesh_utils import Element, get_b_bar, get_geom_limits, get_glob_strain_disp, get_surf_elems, global_dof, read_mesh

# ----------------------------------------
#        Class definitions
# ----------------------------------------

class CruciformDataset(torch.utils.data.Dataset):
    def __init__(self, trials, data_dir, features, outputs, info, transform=None, seq_len=1):
        
        self.trials = trials
        self.data_dir = data_dir
        self.features = features
        self.outputs = outputs
        self.info = info
        self.transform = transform
        self.seq_len = seq_len
        
        self.files = self.get_files()

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, idx):
        
        x, y, t, f, t_pts, n_elems = self.load_data(self.files[idx])
        
        de = torch.diff(x.reshape(t_pts, n_elems,-1).permute(1,0,2), dim=1).reshape(-1, x.shape[-1])
        # de = torch.diff(x.reshape(t_pts, n_elems, -1), dim=0)
        
        # Discarding last time-step due to the use of the strain increment de
        x = x[:-n_elems,:]
        y = y[:-n_elems,:]
        t = t[:-n_elems,:]
        f = f[:-1]

        t = torch.cat([torch.zeros(n_elems, t.shape[-1]), t], 0)
        dt = torch.diff(t.reshape(t_pts,n_elems,-1).permute(1,0,2), dim=1)
        #dt = torch.diff(t.reshape(t_pts, n_elems, -1), dim=0)
        de_dt = torch.cat([torch.zeros(n_elems, x.shape[-1]), x], 0).reshape(t_pts, n_elems,-1).permute(1,0,2)
        #de_dt = torch.cat([torch.zeros(n_elems, x.shape[-1]), x], 0).reshape(t_pts, n_elems,-1)
        #de_dt = torch.diff(de_dt, dim=0)
        de_dt = torch.diff(de_dt, dim=1)
        de_dt /= (dt + 1e-12)
        de_dt = de_dt.reshape(-1,de_dt.shape[-1])

        # Adding padding of zeros to input to make predictions start at zero        
        pad_zeros = torch.zeros((self.seq_len-1) * n_elems, x.shape[-1])
        x = torch.cat([pad_zeros, x], 0)

        x_de = self.rolling_window(x.reshape(t_pts-1 + self.seq_len-1, n_elems,-1), seq_size=self.seq_len)
        x_de = x_de.reshape(-1,*x_de.shape[2:])
        x_de[:,[0,1],:] = x_de[:,[2,3],:]
        x_de[:,2,:] += de
        x_de[:,3,:] -= de
        # x_de[:,:,[0,1],:] = x_de[:,:,[2,3],:]
        # x_de[:,:,2,:] += de
        # x_de[:,:,3,:] -= de
        
        if self.transform != None:
            
            x = self.transform(x)
            x_de = self.transform(x_de)
            
        #x = self.rolling_window(x.reshape(t_pts + self.seq_len-1, n_elems,-1), seq_size=self.seq_len)
        x = self.rolling_window(x.reshape(t_pts-1 + self.seq_len-1, n_elems, -1), seq_size=self.seq_len)
        x = x.reshape(-1,*x.shape[2:])

        #y = y.reshape(t_pts,n_elems,-1).permute(1,0,2)
        y = y.reshape(t_pts-1,n_elems,-1).permute(1,0,2)
        #y = y.reshape(t_pts-1,n_elems,-1)
        y = y.reshape(-1,y.shape[-1])
        
        x_0 = x[::t_pts-1]
        y_0 = y[::t_pts-1]
        #x_0 = x[0]
        #y_0 = y[0]

        idx_t = torch.randperm(t_pts)
        ord_elem = torch.argsort(torch.rand(*x.shape[:2]), dim=-1)
        idx_elem = torch.randperm(x.shape[0])

        ord_t = torch.sort(idx_t).indices
        # # ord_elem = torch.sort(idx_elem).indices
        # x = x[idx_t][torch.arange(x.shape[0]).unsqueeze(-1), ord_elem]
        # y = y[idx_t][torch.arange(y.shape[0]).unsqueeze(-1), ord_elem]
        # x_de = x_de[idx_t][torch.arange(x_de.shape[0]).unsqueeze(-1), ord_elem]
        # de = de[idx_t][torch.arange(de.shape[0]).unsqueeze(-1), ord_elem]
        # de_dt = de_dt[idx_t][torch.arange(de_dt.shape[0]).unsqueeze(-1), ord_elem]

        return x[idx_elem], y[idx_elem], x_de[idx_elem], de[idx_elem], x_0, y_0, de_dt[idx_elem], f, t_pts, n_elems, ord_t, ord_elem 
        #return x, y, x_de, de, x_0, y_0, de_dt, f[idx_t], t_pts-1, n_elems, ord_t, ord_elem   
    
    def rolling_window(self, x, seq_size, step_size=1):
        # Unfold dimension to make sliding window
        return x.unfold(0,seq_size,step_size).permute(1,0,3,2)
        #return x.unfold(0,seq_size,step_size).permute(0,1,3,2)
    
    def get_files(self):

        files = [os.path.join(self.data_dir, trial + '.parquet') for trial in self.trials]
        
        return files
    
    def load_data(self, path):

        # Reading data
        data = pq.ParquetDataset(path)
        data= data.read_pandas(use_threads=True,
                               columns=self.features+self.outputs+['t','id','inc','fxx_t','fyy_t'])
        data= data.to_pandas(self_destruct=True)
        
        # Extracting features, output labels and other info
        x = torch.from_numpy(data[self.features].dropna().values).float()
        y = torch.from_numpy(data[self.outputs].dropna().values).float()
        f = torch.from_numpy(data[['fxx_t','fyy_t']].dropna().values).float()
        t = torch.from_numpy(data['t'].values).float()
        #a = torch.from_numpy(data['area'].dropna().values).float()
        ids = set(data['id'].values)

        t_pts = len(list(set(t.numpy())))
        n_elems = len(ids)

        return x, y, t.unsqueeze(-1), f[::n_elems], t_pts, n_elems
    
# -------------------------------
#       Method definitions
# -------------------------------

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
        m.bias.data.fill_(0.01)
    # elif isinstance(m, torch.nn.GRU):
    #     for n, p in m.named_parameters():
    #         if 'weight_hh' in n:
    #             p.data.reshape(p.shape[0]//p.shape[-1],p.shape[-1],p.shape[-1]).copy_(torch.eye(p.shape[-1])).reshape([-1,p.shape[-1]])
    #         elif 'bias_hh' in n:
    #             p.data.fill_(0.0)

# def batch_jacobian(y,x, var):
    
#     batch = x.size(0)
#     inp_dim = x.size(-1)
#     out_dim = y.size(-1)

#     grad_output = torch.eye(out_dim, device=y.device).unsqueeze(1).repeat(1,batch,1)
#     gradient = torch.autograd.grad(y,x,grad_output,retain_graph=True, create_graph=True, is_grads_batched=True)
#     J = gradient[0][:,:,-1].permute(1,0,2)
    
#     # for i in range(out_dim):
#     #     grad_output = torch.zeros([batch,out_dim])
#     #     grad_output[:,i] = 1

#     #     gradient = torch.autograd.grad(y,x,grad_output,retain_graph=True, create_graph=True)
#     #     J[:,i,:] = gradient[0][:,-1]
#     #     #print("hey")
    
#     return J*(1/var)

def batch_jacobian(model, x, var):
    
    batch = x.size(0)
    inp_dim = x.size(-1)

    grad_output = torch.eye(inp_dim, device='cpu').unsqueeze(1).repeat(1,batch,1)

    _, vjp_fn = torch.func.vjp(model, x)
    
    J = torch.func.vmap(vjp_fn)(grad_output)[0][:,:,-1].permute(1,0,2)
    J *= (1/var)
    
    # gradient = torch.autograd.grad(y,x,grad_output,retain_graph=True, create_graph=True, is_grads_batched=True)
    # J = gradient[0][:,:,-1].permute(1,0,2)
    
    return J

def optimizer_to(optim, device):
    # move optimizer to device
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)

def train_loop(dataloader, model, l_total, l_functions, optimizer):
    '''
    Custom loop for neural network training, using mini-batches

    '''
    global steps

    for _, l_fn in l_functions.items():
        l_fn.reset_stats()

    data_iter = iter(dataloader)

    num_batches = len(dataloader)

    num_losses = len(l_functions.keys())

    next_batch = data_iter.__next__() # start loading the first batch
    next_batch = [ _.cuda(non_blocking=True) for _ in next_batch ]

    logs = {k: None for k in l_functions.keys()}
    logs.update({'lambdas': torch.zeros(num_batches, num_losses)})

    model.train()

    for batch_idx in range(len(dataloader)):

        # Extracting variables for training
        X_train, y_train, x_de, de, x_0, y_0, de_dt, f, t_pts, n_elems, ord_t, ord_elem = next_batch

        if batch_idx + 1 != len(dataloader): 
            # start copying data of next batch
            next_batch = data_iter.__next__()
            next_batch = [_.cuda(non_blocking=True) for _ in next_batch]
        
        X_train = X_train.squeeze(0)
        y_train = y_train.squeeze(0)
        x_de = x_de.squeeze(0)
        de = de.squeeze(0)
        x_0 = x_0.squeeze(0)
        y_0 = y_0.squeeze(0)
        de_dt = de_dt.squeeze(0)
        f = f.squeeze(0)
        ord_elem = ord_elem.squeeze(0)

        x_batches = X_train.split(X_train.shape[0] // BATCH_DIVIDER)[:-1]
        y_batches = y_train.split(y_train.shape[0] // BATCH_DIVIDER)[:-1]
        x_de_batches = x_de.split(x_de.shape[0] // BATCH_DIVIDER)[:-1]
        de_batches = de.split(de.shape[0] // BATCH_DIVIDER)[:-1]
        de_dt_batches = de_dt.split(de_dt.shape[0] // BATCH_DIVIDER)[:-1]
        f_batches = f.split(f.shape[0] // BATCH_DIVIDER)[:-1]

        # x_batches = X_train.split(X_train.shape[0] // (t_pts // 10))
        # y_batches = y_train.split(y_train.shape[0] // (t_pts // 10))
        # x_de_batches = x_de.split(x_de.shape[0] // (t_pts // 10))
        # de_batches = de.split(de.shape[0] // (t_pts // 10))
        # de_dt_batches = de_dt.split(de_dt.shape[0] // (t_pts // 10))
        # f_batches = f.split(f.shape[0] // (t_pts // 10))

        # ord_elems = ord_elem.split(ord_elem.shape[0] // (t_pts // 10))

        running_losses = dict.fromkeys(l_functions.keys(), 0.0)
        running_losses.update({'lambdas': torch.zeros(num_losses)})
            
        for i, batch in enumerate(x_batches):
            
            losses = dict.fromkeys(l_functions.keys())

            x = batch
            #x = batch.reshape(-1,*batch.shape[2:]).requires_grad_(True)
            xde = x_de_batches[i].reshape_as(x).requires_grad_(True)
            
            with torch.autocast(device_type='cuda', dtype=torch.float32, enabled=USE_AMP):
                
                if config.model.cholesky:
                    l = model(x)
                    pred = (l @ (x[:,-1] * std_ + mean_).unsqueeze(-1)).squeeze(-1)

                    if 'clausius' in losses.keys():
                        l_2 = model(xde)
                        pred_2 = (l_2 @ (xde[:,-1] * std_ + mean_).unsqueeze(-1)).squeeze(-1)

                else:

                    if 'normalization' in losses.keys():
                        pred_0 = model(x_0)
                    if 'clausius' in losses.keys():
                        pred_2 = model(xde)
                    
                    pred = model(x)  
            
            with torch.autocast(device_type='cuda', dtype=torch.float32, enabled=USE_AMP):
                for j, (k, _) in enumerate(losses.items()):
                
                    if k == 'triax' or k == 'lode':
                        losses[k] = l_functions.get(k)(pred)
                    elif k == 'clausius':
                        losses[k] = l_functions.get(k)(pred, pred_2, de_batches[i].reshape_as(pred))
                    elif k == 'data':
                        losses[k] = l_functions.get(k)(pred, y_batches[i])
                    elif k == 'normalization':
                        losses[k] = l_functions.get(k)(pred_0)
                    elif k == 'p_power':
                        losses[k] = l_functions.get(k)(pred, de_dt_batches[i])
                    elif k == 'surf_trac':
                        losses[k] = l_functions.get(k)(pred, f_batches[i], TRAC_SURF, n_elems.item(), ord_elem.squeeze())
                    elif k == 's_grad':
                        losses[k] = l_functions.get(k)(y_batches[i].reshape_as(pred), B_BAR, n_elems, ord_elems[i])

                loss = l_total(losses, steps)

            scaler.scale(loss).backward()

            g_norm_logger.log_g_norm(model)
        
            if ((i + 1) % ITERS_TO_ACCUMULATE == 0) or (i + 1 == len(x_batches)):    
                
                scaler.step(optimizer)
                old_scaler = scaler.get_scale()
                scaler.update()
                new_scaler = scaler.get_scale()
                # Added to prevent UserWarning: Detected call of `lr_scheduler.step()` before `optimizer.step()`
                if new_scaler >= old_scaler:
                    warmup_lr.step()
                
                optimizer.zero_grad(set_to_none=True)            
            
            running_losses['lambdas'] += l_total.lmbda_ema.clone().detach().cpu()
          
            steps += 1
        
        logs['lambdas'][batch_idx,:] = running_losses['lambdas'] / (i+1)

        print('\r> Train: %d/%d' % (batch_idx + 1, num_batches), end='')

    for k, v in logs.items():
        if k == 'lambdas':
            logs.update({k: torch.mean(v,0)})
        else:
            logs.update({k: l_functions[k].report_loss()})
        
    #torch.cuda.empty_cache()

    return logs

def test_loop(dataloader, model, l_fn):

    l_fn.reset_stats()

    data_iter = iter(dataloader)

    num_batches = len(dataloader)
    
    model.eval()

    with torch.no_grad():

        for batch_idx in range(len(dataloader)):
        
            X_test, y_test, _, _, _, _, _, _, t_pts, _, _, _ = data_iter.__next__()
        
            X_test = X_test.squeeze(0).to(DEVICE)
            y_test = y_test.squeeze(0).to(DEVICE)

            x_batches = X_test.split(X_test.shape[0] // BATCH_DIVIDER)[:-1]
            y_batches = y_test.split(y_test.shape[0] // BATCH_DIVIDER)[:-1]

            # x_batches = X_test.split(X_test.shape[0] // t_pts)
            # y_batches = y_test.split(y_test.shape[0] // t_pts)

            for i, batch in enumerate(x_batches):
               
                with torch.autocast(device_type='cuda', dtype=torch.float32, enabled=USE_AMP):
                    
                    if config.model.cholesky:
                        l = model(batch)
                        # l = model(batch.reshape(-1,*batch.shape[2:]))
                        # pred = (l @ (batch.reshape(-1,*batch.shape[2:])[:,-1] * std_ + mean_).unsqueeze(-1)).squeeze(-1)
                        pred = (l @ (batch[:,-1] * std_ + mean_).unsqueeze(-1)).squeeze(-1)
                    else:
                        #pred = model(batch.reshape(-1,*batch.shape[2:])) # stress rate
                        pred = model(batch)
            
                    test_loss = l_fn(pred, y_batches[i].reshape_as(pred))

            print('\r> Test: %d/%d' % (batch_idx + 1, num_batches), end='')

    #torch.cuda.empty_cache()

    return l_fn.report_loss()

if __name__ == '__main__':
    
    # Loading configuration file
    if len(sys.argv) == 2:
        CONFIG_PATH = str('\\'.join(sys.argv[1].split('\\')[:-1]))
        CONFIG_NAME = str(sys.argv[1].split('\\')[-1])
    else:
        CONFIG_PATH = './config/'
        CONFIG_NAME = 'config_direct.yaml'

    # config = load_config(CONFIG_PATH, 'config_direct.yaml')
    config = load_config(CONFIG_PATH, CONFIG_NAME)

# ---------------------------------------------------------------
#                     Pytorch Configurations
# ---------------------------------------------------------------
    os.environ['CUDA_LAUNCH_BLOCKING'] = "0"
    
    # Disabling Debug APIs
    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)
    torch.autograd.profiler.emit_nvtx(False)

    # Default floating point precision for pytorch
    torch.set_default_dtype(torch.float32)

    if torch.cuda.is_available():  
        dev = "cuda:0"
        KWARGS = {'num_workers': 0, 'pin_memory': True} 
        
        # Cleaning GPU cache
        torch.cuda.empty_cache()
    else:  
        dev = "cpu"  
        KWARGS = {'num_workers': 0}
    
    DEVICE = torch.device(dev)

    # Automatic mixed precision
    USE_AMP = config.train.use_amp

# ---------------------------------------------------------------

    # Specifying random seed
    os.environ['PYTHONHASHSEED']=str(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

# ---------------------------------------------------------------
#                     Defining constants
# ---------------------------------------------------------------  
    
    ITERS_TO_ACCUMULATE = config.train.iters_to_accumulate

    BATCH_DIVIDER = config.train.batch_divider

# ---------------------------------------------------------------
#                     WANDB Configurations
# ---------------------------------------------------------------    

    if config.wandb.log:

        # WANDB model details
        WANDB_CONFIG = {
            'inputs': len(config.data.inputs),
            'outputs': len(config.data.outputs),
            'hidden_layers': len(config.model.hidden_size) if type(config.model.hidden_size) is list else 1,
            'hidden_units': f'{*config.model.hidden_size,}' if type(config.model.hidden_size) is list else config.model.hidden_size,
            'stack_units': config.model.num_layers,        
            'epochs': config.train.epochs,
            'l_rate': config.train.l_rate,
            'l2_reg': config.train.w_decay
        }

        tags = [f'alpha_{config.train.loss_settings.alpha}', f'beta_{config.train.loss_settings.beta}', f'tau_{config.train.loss_settings.tau}', f'batch_{BATCH_DIVIDER}', f'cholesky_{config.model.cholesky}'] + list(config.train.losses.values()) + [f'{a[0]}_bias_{a[1]}' for a in list(zip(config.model.bias.keys(),config.model.bias.values()))]

        # Starting WANDB logging
        WANDB_RUN = wandb.init(project=config.wandb.project, 
                               entity=config.wandb.entity, 
                               config=WANDB_CONFIG,
                               tags=tags)
        
        # Tagging the model
        MODEL_TAG = WANDB_RUN.name

        if config.telegram.notify:
            print('')

    else:

        # Tagging the model
        MODEL_TAG = time.strftime("%Y%m%d-%H%M%S")

# ---------------------------------------------------------------
#                   Configuring directories
# ---------------------------------------------------------------
    
    # Temp folder for model checkpoint
    TEMP_DIR = os.path.join(config.dirs.temp, MODEL_TAG)
    
    # Output paths
    DIR_PROJECT = os.path.join('outputs', config.wandb.project)
    DIR_RUN = os.path.join(DIR_PROJECT, 'models', MODEL_TAG)
    DIR_VAL = os.path.join(DIR_PROJECT, 'val')    

    SCALER_FILE = os.path.join(DIR_RUN, 'scaler.pkl')
    MODEL_FILE = os.path.join(DIR_RUN, 'model.pt')
    CONF_FILE = os.path.join(DIR_RUN, 'config.yaml')
    
    # Creating output directories
    directories = [TEMP_DIR, DIR_PROJECT, DIR_RUN, DIR_VAL]

    for dir in directories:
       
        os.makedirs(dir, exist_ok=True)
       
# ---------------------------------------------------------------

    # Splitting training dataset
    train_trials, test_trials = train_test_split(config.dirs.trials, config.data.split)

    if config.data.normalize.type != None:

        # Gathering statistics on training dataset
        stat_vars = get_data_stats(config.dirs.train, train_trials, ['tag'] + config.data.inputs, config.data.normalize.type)

        # Defining data transforms - normalization and noise addition 
        transform = transforms.Compose([
            get_data_transform(config.data.normalize.type, stat_vars)
        ])

        if config.model.cholesky:
            std_ = stat_vars['std'].to(DEVICE)
            mean_ = stat_vars['mean'].to(DEVICE)

    # Preparing dataloaders for mini-batch training
    train_dataset = CruciformDataset(train_trials, 
                                     config.dirs.train, 
                                     config.data.inputs, 
                                     config.data.outputs, 
                                     config.data.info, 
                                     transform=transform, 
                                     seq_len=config.data.seq_len)

    test_dataset = CruciformDataset(test_trials, 
                                    config.dirs.train, 
                                    config.data.inputs, 
                                    config.data.outputs, 
                                    config.data.info, 
                                    transform=transform, 
                                    seq_len=config.data.seq_len)
    
    train_dataloader = DataLoader(train_dataset, shuffle=True, **KWARGS)
    test_dataloader = DataLoader(test_dataset, **KWARGS)

    # model = GRUModel(input_dim=len(config.data.inputs),
    #                  output_dim=len(config.data.outputs), 
    #                  hidden_dim=config.model.hidden_size, 
    #                  layer_dim=config.model.num_layers)

    model = GRUModelCholesky(input_dim=len(config.data.inputs),
                             output_dim=len(config.data.outputs), 
                             hidden_dim=config.model.hidden_size, 
                             layer_dim=config.model.num_layers,
                             cholesky=config.model.cholesky,
                             fc_bias=config.model.bias.fc,
                             gru_bias=config.model.bias.gru,
                             attention=config.model.attention, 
                             device=DEVICE)

    model.apply(init_weights)
    model.to(DEVICE, non_blocking=True)

    optimizer = torch.optim.AdamW(params=model.parameters(),
                                  weight_decay=config.train.w_decay, 
                                  lr=config.train.l_rate.max)
    
    if config.train.use_amp:
        # Creates a GradScaler
        scaler = torch.cuda.amp.GradScaler()
    
    if config.train.grad_clip.use_clip:
       
        optimizer = QuantileClip.as_optimizer(optimizer=optimizer, 
                                              quantile=config.train.grad_clip.quantile, 
                                              history_length=config.train.grad_clip.hist_length,
                                              global_threshold=config.train.grad_clip.global_clip,
                                              amp_scaler=scaler)
       
    steps_warmup = config.train.l_rate.warmup_steps
    
    steps_annealing = (config.train.epochs - (steps_warmup // (len(train_dataloader)*config.train.batch_divider // ITERS_TO_ACCUMULATE))) * (len(train_dataloader) * config.train.batch_divider // ITERS_TO_ACCUMULATE)

    # milestones = [steps_annealing//13]

    # scheduler_last = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
    #                                                             T_max=steps_annealing//5-milestones[0],
    #                                                             eta_min=1e-4)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                           steps_annealing, 
                                                           eta_min=config.train.l_rate.min)
    
    # chain_schedules = torch.optim.lr_scheduler.SequentialLR(optimizer=optimizer,
    #                                                         schedulers=[scheduler,scheduler_last],
    #                                                         milestones=milestones)
    
    warmup_lr = GradualWarmupScheduler(optimizer, 
                                       multiplier=1, 
                                       total_epoch=steps_warmup,
                                       after_scheduler=scheduler)
    
    # Initializing the early_stopping object
    early_stopping = EarlyStopping(patience=config.train.e_stop.patience, 
                                   path=config.train.e_stop.checkpoint, 
                                   delta=config.train.e_stop.delta, 
                                   verbose=True)

    # Initializing loss functions
    loss_func_names = config.train.losses
    loss_functions = {config.train.losses.get(k): instantiate_losses(v) for k,v in loss_func_names.items()}

    l_total = Relobralo(params=model.parameters(), 
                        num_losses=len(loss_functions.keys()),
                        alpha=config.train.loss_settings.alpha,
                        beta=config.train.loss_settings.beta,
                        tau=config.train.loss_settings.tau)
    
    l_test = DataLoss()

    g_norm_logger = GradNormLogger(amp_scaler=scaler)

    if 's_grad' or 'surf_trac' in loss_func_names.values():
        # Reading mesh file
        MESH, CONNECTIVITY, DOF = read_mesh(config.dirs.mesh)

        # Defining geometry limits
        X_MIN, X_MAX, Y_MIN, Y_MAX = get_geom_limits(MESH)

        # Element centroids
        CENTROIDS = pd.read_csv(os.path.join(config.dirs.mesh,'centroids.csv'), usecols=['cent_x','cent_y']).values

        # Defining edge boundary conditions:
        #   0 - no constraint
        #   1 - displacements fixed along the edge
        #   2 - displacements constant along the edge
        BC_SETTINGS = {

            'b_conds': {
                'left': {
                    'cond': [1,0],
                    'dof': global_dof(MESH[MESH[:,1]==X_MIN][:,0]),
                    'm_dof': global_dof(MESH[(MESH[:,1]==X_MIN) & (MESH[:,2]==Y_MIN)][:,0]),
                    'nodes': MESH[MESH[:,1]==X_MIN]
                },  
                'bottom': {
                    'cond': [0,1],
                    'dof': global_dof(MESH[MESH[:,-1]==Y_MIN][:,0]),
                    'm_dof': global_dof(MESH[(MESH[:,1]==X_MIN) & (MESH[:,2]==Y_MIN)][:,0]),
                    'nodes': MESH[MESH[:,2]==Y_MIN]
                },
                'right': {
                    'cond': [2,0],
                    'dof': global_dof(MESH[MESH[:,1]==X_MAX][:,0]),
                    'm_dof': global_dof(MESH[(MESH[:,1]==X_MAX) & (MESH[:,2]==Y_MAX/2)][:,0]),
                    'nodes': MESH[MESH[:,1]==X_MAX]
                },
                'top': {
                    'cond': [0,2],
                    'dof': global_dof(MESH[MESH[:,-1]==Y_MAX][:,0]),
                    'm_dof': global_dof(MESH[(MESH[:,1]==X_MAX/2) & (MESH[:,2]==Y_MAX)][:,0]),
                    'nodes': MESH[MESH[:,2]==Y_MAX]
                }
            }
        }
        
        # Total degrees of freedom
        TOTAL_DOF = MESH.shape[0] * 2

        # Global degrees of freedom
        GLOBAL_DOF = list(range(TOTAL_DOF)) 

        # Constructing element properties based on mesh info
        ELEMENTS = [Element(CONNECTIVITY[i,:], MESH[CONNECTIVITY[i,1:]-1,1:], DOF[i,:]) for i in range(CONNECTIVITY.shape[0])]

        TRAC_SURF = {
            'top': get_surf_elems(MESH, CONNECTIVITY, BC_SETTINGS['b_conds'], side='top'),
            'right': get_surf_elems(MESH, CONNECTIVITY, BC_SETTINGS['b_conds'], side='right')
        }

        # Global strain-displacement matrix
        B_GLOB = get_glob_strain_disp(ELEMENTS, TOTAL_DOF, BC_SETTINGS)

        B_BAR, ACTIVE_DOF = get_b_bar(BC_SETTINGS, B_GLOB, GLOBAL_DOF)

        B_BAR = torch.from_numpy(B_BAR).float().to(DEVICE)

        del B_GLOB

    if config.wandb.log:
        wandb_log_dict = dict.fromkeys(loss_functions.keys(), 0.0)
        wandb_log_dict.update({f'lambda_{k+1}': 0.0 for k in range(len(loss_func_names))})
        wandb_log_dict.update({'test': 0.0, 'g_norm': 0.0})
        wandb.watch(model, log='all')

    log_dict = {k: {} for k in loss_functions.keys()}
    log_dict.update({'epoch': {}, 'test': {}, 'lambdas': {}, 'g_norm': {}})
    
    steps = 0

    for t in range(config.train.epochs):

        g_norm_logger.reset_stats()

        print('\r--------------------\nEpoch [%d/%d]\n' % (t + 1, config.train.epochs))

        log_dict['epoch'][t] = t + 1

        # Train loop
        start_train = time.time()

        # if steps >= milestones[0]:
        #     scheduler_last.base_lrs = scheduler.get_last_lr()

        #--------------------------------------------------------------
        #logs = train_loop(train_dataloader, model, l_total, l_data, l_cd, l_0, optimizer)
        logs = train_loop(train_dataloader, model, l_total, loss_functions, optimizer)
        #--------------------------------------------------------------

        for k, _ in log_dict.items():
            if k != 'lambdas':
                if k == 'epoch' or k == 'test' :
                    pass
                elif k == 'g_norm':
                    log_dict[k][t] = g_norm_logger.get_g_norm()
                else:
                    log_dict[k][t] = logs[k]
            else:
                log_dict[k][t] = logs[k].tolist()
    
        end_train = time.time()

        print('. lr: {:.4e} | g_norm: {:0.4e} -- {:.3f}s \n'.format(warmup_lr.get_lr()[0], g_norm_logger.get_g_norm(), end_train - start_train))
         
        # Test loop
        start_test = time.time()

        #-----------------------------------------------------------------------------------
        v_loss = test_loop(test_dataloader, model, l_test)
        #-----------------------------------------------------------------------------------

        log_dict['test'][t] = v_loss

        end_test = time.time()

        print(' -- %.3fs' % (end_test - start_test))

        str_ = '\n'
        str_ += f"\t> {list(filter(lambda a: 'data' in a, log_dict.keys()))[0]:<13} \t{log_dict['data'][t]:.6e}\n"
        str_ += f"\t> {list(filter(lambda a: 'test' in a, log_dict.keys()))[0]:<13} \t{log_dict['test'][t]:.6e}\n\n"
        
        for k,v in log_dict.items():
            if k in loss_func_names.values():

                if (k == 'data') or (k == 'test'):
                    pass
                else:
                    str_ += f'\t> {k:<13} \t{v[t]:.6e}\n'
            
            elif k == 'lambdas':
                str_ += f'\n\t> {k:<13} \t' + ' '.join(f"{l:.3f}" for l in v[t])
        
        print(str_)

        if t > config.train.e_stop.start_at:
            early_stopping(log_dict['test'][t], log_dict['data'][t], model)

        if config.wandb.log:
            for k, _ in wandb_log_dict.items():
                if ('lambda' not in k):
                    if (k != 'epoch') and (k != 'l_rate'):
                        wandb_log_dict[k] = log_dict[k][t]
    
            for l in range(len(loss_func_names)):
                wandb_log_dict.update({f'lambda_{l+1}': log_dict['lambdas'][t][l]})
            
            wandb_log_dict.update( {'epoch': t, 'l_rate': warmup_lr.get_lr()[0]} )
            
            wandb.log(wandb_log_dict)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break
    
    print("Done!")

    # Load the last checkpoint with the best model
    model.load_state_dict(torch.load(config.train.e_stop.checkpoint))

    # Save model state
    torch.save(model.state_dict(), os.path.join(DIR_RUN, 'model.pt'))
    
    # Save data scaler
    joblib.dump(stat_vars, os.path.join(DIR_RUN, 'scaler.pkl'))

    # Save config
    save_config(config, CONF_FILE)

    # At the end of training, save the model artifact
    if config.wandb.log:
        
        # Name this artifact after the current run
        model_artifact_name = WANDB_RUN.id + '_' + MODEL_TAG
        # Create a new artifact
        model_artifact = wandb.Artifact(model_artifact_name, type='model')
        # Add files to the artifact
        model_artifact.add_file(MODEL_FILE)
        model_artifact.add_file(SCALER_FILE)
        model_artifact.add_file(CONF_FILE)
        # Log the model to W&B
        WANDB_RUN.log_artifact(model_artifact)
        # Call finish if you're in a notebook, to mark the run as done
        WANDB_RUN.finish()

    # Deleting temp folders
    try:
        shutil.rmtree(config.dirs.temp)
        if config.wandb.log:
            shutil.rmtree('wandb')
    except (FileNotFoundError, PermissionError):
        pass
    

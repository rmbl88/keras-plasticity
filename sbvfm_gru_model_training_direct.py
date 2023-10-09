# ---------------------------------
#    Library and function imports
# ---------------------------------

import cProfile, pstats
from glob import glob
from msvcrt import kbhit
import os
import shutil
import joblib
# from pytools import F
from constants import *
from functions import (
    CoVWeightingLoss,
    GRUModel,
    batch_jacobian,
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
# import pytorch_warmup as warmup
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
        self.data = [pq.ParquetDataset(file).read_pandas(columns=self.features+self.outputs+['t','id']).to_pandas() for file in self.files]

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, idx):
        
        x = torch.from_numpy(self.data[idx][self.features].dropna().values).float()
        e = torch.from_numpy(self.data[idx][self.features].dropna().values).float()
        y = torch.from_numpy(self.data[idx][self.outputs].dropna().values).float()
        t = torch.from_numpy(self.data[idx]['t'].values).float()
        
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
        e = e.reshape(t_pts,n_elems,-1).permute(1,0,2)
        de = torch.zeros_like(e)
        de[:,1:] = torch.diff(e,dim=1)
        x = x.reshape(-1,*x.shape[2:])
        de = de.reshape(-1,de.shape[-1])
        #t = self.rolling_window(t.reshape(t_pts,n_elems,-1), seq_size=self.seq_len)

        #y = y.reshape(t_pts-1,n_elems,-1)[self.seq_len-1:].reshape(-1,y.shape[-1])
        #y = y.reshape(t_pts,n_elems,-1)[self.seq_len-1:].permute(1,0,2)
        y = y.reshape(t_pts,n_elems,-1).permute(1,0,2)
        dy = torch.zeros_like(y)
        dy[:,1:] = torch.diff(y, dim=1)
        y = y.reshape(-1,y.shape[-1])
        dy = dy.reshape(-1,dy.shape[-1])

        idx_ = torch.randperm(x.shape[0])

        return x[idx_], y[idx_], t, t_pts, n_elems, de[idx_], dy[idx_]     
    
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

# def batch_jacobian(f, x):
#     f_sum = lambda x: torch.sum(f(x[:,-1]), axis=0)
#     return jacobian(f_sum, x[:,-1],create_graph=True).permute(1,0,2)

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

        num_batches = len(dataloader)
        
        logs = {
            'loss': {},
            'jac_loss': {},
            's_loss': {},
            'xy_loss': {}
        }
               
        model.train()
               
        for batch_idx in range(len(dataloader)):

            # Extracting variables for training
            X_train, y_train, _, t_pts, n_elems, de, dy = data_iter.__next__()
            
            X_train = X_train.squeeze(0).to(device)
            y_train = y_train.squeeze(0).to(device)
            de = de.squeeze(0).to(device)
            dy = dy.squeeze(0).to(device)

            x_batches = X_train.split(X_train.shape[0]//32)
            y_batches = y_train.split(y_train.shape[0]//32)
            de_batches = de.split(de.shape[0]//32)
            dy_batches = dy.split(dy.shape[0]//32)

            running_loss = 0.0
            jac_loss = 0.0
            s_loss = 0.0
            xy_loss = 0.0
            
            for i, batch in enumerate(x_batches):
                x = batch.requires_grad_(True)
                with torch.autocast(device_type='cuda', dtype=torch.float32, enabled=USE_AMP):
                    pred = model(x)

                a = batch_jacobian(pred.cpu(),x.cpu())
                s = pred[:,:3]  # stress
                # l = pred[:,3:]  # cholesky factors
                # L = torch.zeros(l.size(0),3,3).to(device)
                # tril_indices = torch.tril_indices(row=3, col=3, offset=0)
                # L[:,tril_indices[0], tril_indices[1]] = l  # vector to lower triangular matrix

                # J = L@L.transpose(2,1)  # Jacobian
                # ds = (J@de_batches[i].unsqueeze(-1)).squeeze(-1)          
                                
                # a = J[:,[0,1,2,2],[2,2,0,1]]
                with torch.autocast(device_type='cuda', dtype=torch.float32, enabled=USE_AMP):

                    l_jac = l_fn(ds,dy_batches[i])

                    l_j = l_fn(a, torch.zeros_like(a))

                    l_s = l_fn(s, y_batches[i])

                    loss = l_s + l_jac + l_j
                                    
                scaler.scale(loss/ITERS_TO_ACCUMULATE).backward()
            
                if ((i + 1) % ITERS_TO_ACCUMULATE == 0) or (i + 1 == len(x_batches)):    
                    
                    scaler.step(optimizer)
                    scaler.update()
                    
                    optimizer.zero_grad(set_to_none=True)

                    warmup_lr.step()
            
                # Saving loss values
                running_loss += loss.detach()
                jac_loss += l_jac.detach()
                s_loss += l_s.detach()
                xy_loss += l_j.detach()

                torch.cuda.empty_cache()

            logs['loss'][batch_idx] = running_loss / len(x_batches)
            logs['jac_loss'][batch_idx] = jac_loss / len(x_batches)
            logs['s_loss'][batch_idx] = s_loss / len(x_batches)
            logs['xy_loss'][batch_idx] = xy_loss / len(x_batches)

            print('\r> Train: %d/%d' % (batch_idx + 1, num_batches), end='')

        logs = [np.fromiter(v.values(),dtype=np.float32) for k,v in logs.items()]      
        #-----------------------------
        return logs

    def test_loop(dataloader, model, l_fn):

        data_iter = iter(dataloader)

        num_batches = len(dataloader)
        test_losses = []

        model.eval()

        with torch.no_grad():

            for batch_idx in range(len(dataloader)):
            
                X_test, y_test, _, t_pts, n_elems, de, dy = data_iter.__next__()
            
                X_test = X_test.squeeze(0).to(device)
                y_test = y_test.squeeze(0).to(device)
                de = de.squeeze(0).to(device)
                dy = dy.squeeze(0).to(device)

                x_batches = X_test.split(X_test.shape[0]//32)
                y_batches = y_test.split(y_test.shape[0]//32)
                de_batches = de.split(de.shape[0]//32)
                dy_batches = dy.split(dy.shape[0]//32)

                running_loss = 0.0

                for i, batch in enumerate(x_batches):
                    #model.init_hidden(batch.size(0),device)
                    with torch.autocast(device_type='cuda', dtype=torch.float32, enabled=USE_AMP):
                        pred = model(batch) # stress rate
                    
                    s = pred[:,:3]  # stress
                    l = pred[:,3:]  # cholesky factors
                    L = torch.zeros(l.size(0),3,3).to(device)
                    tril_indices = torch.tril_indices(row=3, col=3, offset=0)
                    L[:,tril_indices[0], tril_indices[1]] = l  # lower triangular matrix to full matrix

                    J = L@L.transpose(2,1)  # Jacobian
                    ds = (J@de_batches[i].unsqueeze(-1)).squeeze(-1)
                
                    with torch.autocast(device_type='cuda', dtype=torch.float32, enabled=USE_AMP):
                        
                        #l_tuples = [(pred[:,0], y_test[:,0]), (pred[:,1], y_test[:,1]), (l_triax,)]
                    
                        #test_loss = l_fn(l_tuples)
                        # test_loss = l_fn(pred[:,0], y_test[:,0]) + l_fn(pred[:,1], y_test[:,1])
                        test_loss = l_fn(s, y_batches[i]) + l_fn(ds, dy_batches[i])
                
                    running_loss += test_loss.item()

                print('\r> Test: %d/%d' % (batch_idx + 1, num_batches), end='')

            test_losses.append(running_loss / len(x_batches))

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

    df_list = []

    dir = os.path.join(TRAIN_MULTI_DIR,'processed/')

    file_list = glob.glob(os.path.join(dir, f'*.parquet'))

    df_list = [pq.ParquetDataset(file).read_pandas(columns=['tag']+FEATURES).to_pandas() for file in tqdm(file_list,desc='Importing dataset files',bar_format=FORMAT_PBAR)]

    raw_data = pd.concat(df_list)
    input_data = raw_data[raw_data['tag'].isin(train_trials)].drop('tag',1).dropna()

    #min = torch.min(torch.from_numpy(input_data.values).float(),0).values
    #max = torch.max(torch.from_numpy(input_data.values).float(),0).values
    std, mean = torch.std_mean(torch.from_numpy(input_data.values.astype(np.float32)),0)

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
    WANDB_LOG = False

    model_1 = GRUModel(input_dim=N_INPUTS, hidden_dim=N_UNITS, layer_dim=H_LAYERS, output_dim=N_OUTPUTS)

    model_1.apply(init_weights)
    model_1.to(device)

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
    
    # Creates a GradScaler once at the beginning of training.
    scaler = torch.cuda.amp.GradScaler()

    # Container variables for history purposes
    log_dict = {
        'epoch': {},
        't_loss': {},
        'v_loss': {},
        'jac_loss': {},
        's_loss': {},
        'xy_loss': {},
        'grad_norm': {}
    }

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

        print('\r--------------------\nEpoch [%d/%d]\n' % (t + 1, epochs))

        log_dict['epoch'][t] = t + 1

        # Train loop
        start_train = time.time()

        #--------------------------------------------------------------
        t_loss, jac_loss, s_loss, xy_loss = train_loop(train_dataloader, model_1, l_fn, optimizer,t)
        #--------------------------------------------------------------

        log_dict['t_loss'][t] = np.mean(t_loss) 
        log_dict['jac_loss'][t] = np.mean(jac_loss)
        log_dict['s_loss'][t] = np.mean(s_loss)
        log_dict['xy_loss'][t] = np.mean(xy_loss)
        #train_loss.append(np.mean(batch_losses))
        # f_loss.append(np.mean(f_l))
        # l_loss.append(np.mean(l_))
        # triax_loss.append(np.mean(t_l))

        end_train = time.time()

        #Apply learning rate scheduling if defined
        #try:
            #scheduler.step(train_loss[t])
        print('. t_loss: %.6e -> lr: %.4e -- %.3fs \n\nl_stress: %.4e | l_jac: %.4e | l_xy: %.4e\n' % (log_dict['t_loss'][t], warmup_lr._last_lr[0], end_train - start_train, log_dict['s_loss'][t], log_dict['jac_loss'][t], log_dict['xy_loss'][t]))
        # except:
        #     print('. t_loss: %.6e -> | wm_l: %.4e -- %.3fs' % (train_loss[t], f_loss[t], end_train - start_train))

        # Test loop
        start_test = time.time()

        #-----------------------------------------------------------------------------------
        v_loss = test_loop(test_dataloader, model_1, l_fn)
        #-----------------------------------------------------------------------------------

        log_dict['v_loss'][t] = np.mean(v_loss)
        #val_loss.append(np.mean(batch_val_losses))

        end_test = time.time()

        print('. v_loss: %.6e -- %.3fs' % (log_dict['v_loss'][t], end_test - start_test))

        if t > 200:
            early_stopping(log_dict['v_loss'][t], model_1)

        if WANDB_LOG:
            wandb.log({
                'epoch': t,
                'l_rate': warmup_lr._last_lr[0],
                'train_loss': log_dict['t_loss'][t],
                'test_loss': log_dict['v_loss'][t],
                'l_jac': log_dict['jac_loss'][t],
                'l_stress': log_dict['s_loss'][t],
                'l_xy': log_dict['xy_loss'][t]
            })

        if  early_stopping.early_stop:
            print("Early stopping")
            break
    
    print("Done!")

    # load the last checkpoint with the best model
    model_1.load_state_dict(torch.load(path))

    # epochs_ = np.reshape(np.array(epochs_), (len(epochs_),1))
    # train_loss = np.reshape(np.array(train_loss), (len(train_loss),1))
    # val_loss = np.reshape(np.array(val_loss), (len(val_loss),1))

    # history = pd.DataFrame(np.concatenate([epochs_, train_loss, val_loss], axis=1), columns=['epoch','loss','val_loss'])

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

    #history.to_csv(output_loss + task + '.csv', sep=',', encoding='utf-8', header='true')

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
        task_ = f"{run.id}_{run.name}"
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

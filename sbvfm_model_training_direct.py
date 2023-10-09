# ---------------------------------
#    Library and function imports
# ---------------------------------
import cProfile, pstats
from glob import glob
from msvcrt import kbhit
import os
import shutil
import joblib
from constants import *
from functions import (
    CoVWeightingLoss,
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
    def __init__(self, trials, root_dir, data_dir, features, outputs, info, scaler_x=None, transform=None):
        
        self.trials = trials
        self.root_dir = root_dir
        self.data_dir = data_dir
        self.scaler_x = scaler_x
        self.features = features
        self.outputs = outputs
        self.info = info
        self.transform = transform
        self.files = [os.path.join(self.root_dir, self.data_dir , trial + '.parquet') for trial in self.trials]
        self.data = [pq.ParquetDataset(file).read_pandas(columns=self.features+self.outputs+['t','id','s1','s2','fxx_t','fyy_t','area','sxx_t','syy_t']).to_pandas() for file in self.files]

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, idx):
        
        x = torch.from_numpy(self.data[idx][self.features].dropna().values)
        y = torch.from_numpy(self.data[idx][self.outputs].dropna().values)
        t = torch.from_numpy(self.data[idx]['t'].values)
        e = torch.from_numpy(self.data[idx][['dep_1','dep_2']].dropna().values)
        f = torch.from_numpy(self.data[idx][['fxx_t','fyy_t']].values)
        a = torch.from_numpy(self.data[idx]['area'].values).reshape(-1,1)

        t_pts = len(list(set(t.numpy())))
        n_elems = len(set(self.data[idx]['id'].values))
        
        if self.transform != None:
            
            x = self.transform(x.permute(1,0).unsqueeze(-1))
            
            return x.squeeze(-1).permute(1,0), y, t, e, f, a, t_pts, n_elems
        else:
            return x, y, t, t_pts, n_elems

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

        next_batch = data_iter.__next__()  # start loading the first batch
        next_batch = [_.cuda(non_blocking=True) for _ in next_batch]

        num_batches = len(dataloader)
        
        losses = torch.zeros(num_batches)
        f_loss = torch.zeros_like(losses)
        triax_loss = torch.zeros_like(losses)
        l_loss = torch.zeros_like(losses)
               
        model.train()
        l_fn.to_train()
        
        for batch_idx in range(len(dataloader)):
    
            batch = next_batch

            if batch_idx + 1 != len(dataloader): 
                # start copying data of next batch
                next_batch = data_iter.__next__()
                next_batch = [_.cuda(non_blocking=True) for _ in next_batch]

            # Extracting variables for training
            X_train, y_train, t, e, f, a, t_pts, n_elems = batch
            
            X_train = X_train.squeeze(0).to(device)
            y_train = y_train.squeeze(0).to(device)
            t = t.squeeze(0).to(device)
            e = e.squeeze(0).to(device)
            f = f.squeeze(0).to(device)
            a = a.squeeze(0).to(device)
            
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=USE_AMP):
                pred = model(X_train) # stress rate
               
            dt = torch.diff(t.reshape(t_pts,n_elems,1),dim=0)
            s_princ_ = torch.zeros([t_pts,n_elems,2]).to(device)
            s_princ_[1:,:,:] = pred.reshape(t_pts-1,n_elems,pred.shape[-1])*dt
            s_princ_ = torch.cumsum(s_princ_,0)

            f_ = (torch.sum(s_princ_*a.reshape([t_pts,n_elems,1]),1)/30.0).unsqueeze(1).repeat(1,n_elems,1).reshape(-1,s_princ_.shape[-1])

            
            s_princ_ord = torch.sort(s_princ_,-1,descending=True).values
            triax = (math.sqrt(2)/3)*(s_princ_ord[:,:,0]+s_princ_ord[:,:,1])/(s_princ_ord[:,:,0]-s_princ_ord[:,:,1]+1e-12)
            l_triax = torch.nn.functional.relu(torch.abs(triax.reshape(-1,1))-2/3)
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
                
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=USE_AMP):
                #l_triaxiality = torch.mean(torch.nn.functional.relu(torch.abs(triaxiality)-2/3)
                
                l_tuples = [(pred[:,0], y_train[:,0]), (pred[:,1], y_train[:,1]), (l_triax,)]
                
                loss = l_fn(l_tuples)
            
            scaler.scale(loss/ITERS_TO_ACCUMULATE).backward()
            
            if ((batch_idx + 1) % ITERS_TO_ACCUMULATE == 0) or (batch_idx + 1 == len(train_dataset)):    
                
                scaler.step(optimizer)
                scaler.update()
                   
                optimizer.zero_grad(set_to_none=True)

                warmup_lr.step()

            # Saving loss values
            losses[batch_idx] = loss.detach().item()
            f_loss[batch_idx] = 0.0
            triax_loss[batch_idx] = l_fn.weighted_losses[-1].detach().item()
            l_loss[batch_idx] = sum(l_fn.weighted_losses[:2]).detach().item()        

            print('\r>Train: %d/%d' % (batch_idx + 1, num_batches), end='')
              
        #-----------------------------
        return losses, f_loss, l_loss, triax_loss

    def test_loop(dataloader, model, l_fn):

        data_iter = iter(dataloader)

        next_batch = data_iter.__next__()  # start loading the first batch
        next_batch = [_.cuda(non_blocking=True) for _ in next_batch]

        num_batches = len(dataloader)
        test_losses = torch.zeros(num_batches)

        model.eval()
        l_fn.to_eval()

        with torch.no_grad():

            for batch_idx in range(len(dataloader)):

                batch = next_batch

                if batch_idx + 1 != len(dataloader): 
                    # start copying data of next batch
                    next_batch = data_iter.__next__()
                    next_batch = [_.cuda(non_blocking=True) for _ in next_batch]
            
                X_test, y_test, t, e, f, a, t_pts, n_elems = batch
            
                X_test = X_test.squeeze(0).to(device)
                y_test = y_test.squeeze(0).to(device)
                t = t.squeeze(0).to(device)
                e = e.squeeze(0).to(device)
                f = f.squeeze(0).to(device)
                a = a.squeeze(0).to(device)

                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=USE_AMP):
                    pred = model(X_test) # stress rate
                        
                dt = torch.diff(t.reshape(t_pts,n_elems,1),dim=0)
                s_princ_ = torch.zeros([t_pts,n_elems,2]).to(device)
                s_princ_[1:,:,:] = pred.reshape(t_pts-1,n_elems,pred.shape[-1])*dt
                s_princ_ = torch.cumsum(s_princ_,0)

                f_ = (torch.sum(s_princ_*a.reshape([t_pts,n_elems,1]),1)/30.0).unsqueeze(1).repeat(1,n_elems,1).reshape(-1,s_princ_.shape[-1])
                s_princ_ord = torch.sort(s_princ_,-1,descending=True).values
                triax = (math.sqrt(2)/3)*(s_princ_ord[:,:,0]+s_princ_ord[:,:,1])/(s_princ_ord[:,:,0]-s_princ_ord[:,:,1]+1e-12)
                l_triax = torch.nn.functional.relu(torch.abs(triax.reshape(-1,1))-2/3)
                
                
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
                
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=USE_AMP):
                    
                    l_tuples = [(pred[:,0], y_test[:,0]), (pred[:,1], y_test[:,1]), (l_triax,)]
                
                    test_loss = l_fn(l_tuples)
                
                test_losses[batch_idx] = test_loss.detach().item()
               

                print('\r>Test: %d/%d' % (batch_idx + 1, num_batches), end='')

        return test_losses
#----------------------------------------------------

    # Default floating point precision for pytorch
    torch.set_default_dtype(torch.float64)

    if torch.cuda.is_available():  
        dev = "cuda:0"
        KWARGS = {'num_workers': 0, 'pin_memory': True} 
    else:  
        dev = "cpu"  
        KWARGS = {'num_workers': 0}
    
    device = torch.device(dev) 

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

    FEATURES = ['ep_1_dir','ep_2_dir','dep_1','dep_2','ep_1','ep_2']
    OUTPUTS = ['ds1','ds2']
    INFO = ['tag','inc','t','theta_ep','s1','s2','theta_sp','fxx_t','fyy_t']

    # Gathering statistics on training dataset

    file_list = []
    df_list = []

    dir = os.path.join(TRAIN_MULTI_DIR,'processed/')
    for r, d, f in os.walk(dir):
        for file in f:
            if '.csv' or '.parquet' in file:
                file_list.append(dir + file)

    df_list = [pq.ParquetDataset(file).read_pandas(columns=['tag']+FEATURES).to_pandas() for file in tqdm(file_list,desc='Importing dataset files',bar_format=FORMAT_PBAR)]

    input_data = pd.concat(df_list)
    input_data = input_data[input_data['tag'].isin(train_trials)].drop('tag',1).dropna()
    std, mean = torch.std_mean(torch.from_numpy(input_data.values),0)

    # Cleaning workspace from useless variables
    del df_list
    del file_list
    del input_data
    gc.collect()

    # Defining data transforms - normalization and noise addition 
    transform = transforms.Compose([
        transforms.Normalize(mean.tolist(), std.tolist()),
        #transforms.RandomApply([AddGaussianNoise(0., 1.)],p=0.15)
    ])

    # Preparing dataloaders for mini-batch training

    train_dataset = CruciformDataset(train_trials, TRAIN_MULTI_DIR, 'processed', FEATURES, OUTPUTS, INFO, transform=transform)
    test_dataset = CruciformDataset(test_trials, TRAIN_MULTI_DIR, 'processed', FEATURES, OUTPUTS, INFO, transform=transform)
    
    train_dataloader = DataLoader(train_dataset, shuffle=True,**KWARGS)
    test_dataloader = DataLoader(test_dataset, **KWARGS)
    
    # Model variables
    N_INPUTS = len(FEATURES)
    N_OUTPUTS = len(OUTPUTS)

    N_UNITS = [20,20,20,20]
    H_LAYERS = len(N_UNITS)

    ITERS_TO_ACCUMULATE = 1

    # Automatic mixed precision
    USE_AMP = True

    # WANDB logging
    WANDB_LOG = True

    model_1 = NeuralNetwork(N_INPUTS, N_OUTPUTS, N_UNITS, H_LAYERS)

    model_1.apply(init_weights)
    model_1.to(device)

    # clip_value = 100
    # for p in model_1.parameters():
    #     p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))

    # Training variables
    epochs = 10000

    # Optimization variables
    learning_rate = 0.05
    lr_mult = 1.0

    params = layer_wise_lr(model_1, lr_mult=lr_mult, learning_rate=learning_rate)
    
    l_fn = CoVWeightingLoss(device=device, n_losses=3)

    weight_decay=0.0

    optimizer = torch.optim.AdamW(params=params, weight_decay=weight_decay)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=20, cooldown=8, factor=0.88, min_lr=[params[i]['lr']*0.00005 for i in range(len(params))])
       
    steps_warmup = 360
    steps_annealing = (epochs - (steps_warmup // (len(train_dataloader) // ITERS_TO_ACCUMULATE))) * (len(train_dataloader) // ITERS_TO_ACCUMULATE)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps_annealing, eta_min=1e-7)
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
            "hidden_units": "/".join(str(x) for x in N_UNITS),        
            "epochs": epochs,
            "lr": learning_rate,
            "l2_reg": weight_decay
        }
        run=wandb.init(project="direct_training_principal_inc_crux_cov", entity="rmbl",config=config)
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

        train_loss.append(torch.mean(batch_losses))
        f_loss.append(torch.mean(f_l))
        l_loss.append(torch.mean(l_))
        triax_loss.append(torch.mean(t_l))
        
        alpha1.append(l_fn.alphas[0])
        alpha2.append(l_fn.alphas[1])
        alpha3.append(l_fn.alphas[2])
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

        val_loss.append(torch.mean(batch_val_losses).item())

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
                'f_loss': f_loss[t],
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

    #plot_history(history, output_loss, True, task)

    torch.save(model_1.state_dict(), output_models + task + '.pt')
    
    joblib.dump([std, mean], output_models + task + '-scaler_x.pkl')
    joblib.dump([FEATURES, OUTPUTS, INFO, N_UNITS], output_models + run.name + '-arch.pkl')

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

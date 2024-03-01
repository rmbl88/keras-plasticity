# ---------------------------------
#    Library and function imports
# ---------------------------------
import os
import shutil
import joblib
from functions import (
    GRUModel,
    EarlyStopping,
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
        
        x, y, t_pts, n_elems = self.load_data(self.files[idx])

        de = torch.diff(x.reshape(t_pts, n_elems,-1).permute(1,0,2), dim=1).reshape(-1, x.shape[-1])

        # Discarding last time-step due to the use of the strain increment de
        x = x[:-n_elems,:]
        y = y[:-n_elems,:]

        # Adding padding of zeros to input to make predictions start at zero        
        pad_zeros = torch.zeros((self.seq_len-1) * n_elems, x.shape[-1])
        x = torch.cat([pad_zeros, x], 0)

        x_de = self.rolling_window(x.reshape(t_pts-1 + self.seq_len-1, n_elems,-1), seq_size=self.seq_len)
        x_de = x_de.reshape(-1,*x_de.shape[2:])
        x_de[:,[0,1],:] = x_de[:,[2,3],:]
        x_de[:,2,:] += de
        x_de[:,3,:] -= de
        
        if self.transform != None:
            
            x = self.transform(x)
            x_de = self.transform(x_de)
            
        #x = self.rolling_window(x.reshape(t_pts + self.seq_len-1, n_elems,-1), seq_size=self.seq_len)
        x = self.rolling_window(x.reshape(t_pts-1 + self.seq_len-1, n_elems,-1), seq_size=self.seq_len)
        x = x.reshape(-1,*x.shape[2:])

        #y = y.reshape(t_pts,n_elems,-1).permute(1,0,2)
        y = y.reshape(t_pts-1,n_elems,-1).permute(1,0,2)
        y = y.reshape(-1,y.shape[-1])
        
        x_0 = x[::t_pts-1]
        y_0 = y[::t_pts-1]

        idx_elem = torch.randperm(x.shape[0])

        return x[idx_elem], y[idx_elem], x_de[idx_elem], de[idx_elem], x_0, y_0, t_pts, n_elems    
    
    def rolling_window(self, x, seq_size, step_size=1):
        # Unfold dimension to make sliding window
        return x.unfold(0,seq_size,step_size).permute(1,0,3,2)
    
    def get_files(self):

        files = [os.path.join(self.data_dir, trial + '.parquet') for trial in self.trials]
        
        return files
    
    def load_data(self, path):

        # Reading data
        data = pq.ParquetDataset(path)
        data= data.read_pandas(use_threads=True,
                               columns=self.features+self.outputs+['t','id'])
        data= data.to_pandas(self_destruct=True)
        
        # Extracting features, output labels and other info
        x = torch.from_numpy(data[self.features].dropna().values).float()
        y = torch.from_numpy(data[self.outputs].dropna().values).float()
        t = torch.from_numpy(data['t'].values).float()
        ids = set(data['id'].values)

        t_pts = len(list(set(t.numpy())))
        n_elems = len(ids)

        return x, y, t_pts, n_elems
    
# -------------------------------
#       Method definitions
# -------------------------------

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

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
        m.bias.data.fill_(0.01)
    # elif isinstance(m, torch.nn.GRU):
    #     for n, p in m.named_parameters():
    #         if 'weight_hh' in n:
    #             p.data.reshape(p.shape[0]//p.shape[-1],p.shape[-1],p.shape[-1]).copy_(torch.eye(p.shape[-1])).reshape([-1,p.shape[-1]])
    #         elif 'bias_hh' in n:
    #             p.data.fill_(0.0)

def train_loop(dataloader, model, l_fn, optimizer):
    '''
    Custom loop for neural network training, using mini-batches

    '''
    data_iter = iter(dataloader)

    num_batches = len(dataloader)

    next_batch = data_iter.__next__() # start loading the first batch
    next_batch = [ _.cuda(non_blocking=True) for _ in next_batch ]
    
    logs = {
        'loss': {},
        'clausius': {},
        'consist': {}
    }
            
    model.train()

    running_loss = 0.0
    clausius = 0.0
    l_consist = 0.0
    total_samples = 0.0
    t_i = 0.0

    eps = 5000
    eps_2 = 10

    for batch_idx in range(len(dataloader)):

        # Extracting variables for training
        #X_train, y_train, x_de, de, _, _ = data_iter.__next__()
        X_train, y_train, x_de, de, x_0, y_0, t_pts, _ = next_batch

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

        x_batches = X_train.split(X_train.shape[0] // BATCH_DIVIDER)
        y_batches = y_train.split(y_train.shape[0] // BATCH_DIVIDER)
        x_de_batches = x_de.split(x_de.shape[0] // BATCH_DIVIDER)
        de_batches = de.split(de.shape[0] // BATCH_DIVIDER)
        
        for i, batch in enumerate(x_batches):
            
            x = batch.requires_grad_(True)
            xde = x_de_batches[i].requires_grad_(True)
            
            with torch.autocast(device_type='cuda', dtype=torch.float32, enabled=USE_AMP):
                pred_0 = model(x_0)
                pred = model(x)
                pred_2 = model(xde)
            
                l_clausius = (eps/2)*torch.sum(torch.square(torch.nn.functional.relu(-0.5*torch.sum((pred-pred_2)*(de_batches[i]),-1))))

                l_0 = eps_2*torch.mean(torch.square(pred_0))
                
                loss = l_fn(pred, y_batches[i]) + l_clausius + l_0
                                
            scaler.scale(loss/ITERS_TO_ACCUMULATE).backward()
        
            if ((i + 1) % ITERS_TO_ACCUMULATE == 0) or (i + 1 == len(x_batches)):    
                
                scaler.step(optimizer)
                old_scaler = scaler.get_scale()
                scaler.update()
                new_scaler = scaler.get_scale()
                # Added condition to prevent UserWarning: Detected call of `lr_scheduler.step()` before `optimizer.step()`
                if new_scaler >= old_scaler:
                    warmup_lr.step()

                optimizer.zero_grad(set_to_none=True)
        
            # Saving loss values
            running_loss += loss.detach() * x.size()[0]
            l_consist += l_0.detach() * x_0.size()[0]
            clausius += l_clausius.detach() * 2/eps
            total_samples += x.size()[0]
            t_i += x_0.size()[0]
            #torch.cuda.empty_cache()

        print('\r> Train: %d/%d' % (batch_idx + 1, num_batches), end='')

    logs['loss'] = running_loss / total_samples
    logs['clausius'] = clausius
    logs['consist'] = l_consist / t_i
    
    torch.cuda.empty_cache()
    return logs

def test_loop(dataloader, model, l_fn):

    data_iter = iter(dataloader)

    num_batches = len(dataloader)
    
    test_loss = 0.0

    model.eval()

    with torch.no_grad():

        running_loss = 0.0
        total_samples = 0.0

        for batch_idx in range(len(dataloader)):
        
            X_test, y_test, _, _, _, _, _, _ = data_iter.__next__()
        
            X_test = X_test.squeeze(0).to(DEVICE)
            y_test = y_test.squeeze(0).to(DEVICE)

            x_batches = X_test.split(X_test.shape[0] // BATCH_DIVIDER)
            y_batches = y_test.split(y_test.shape[0] // BATCH_DIVIDER)

            for i, batch in enumerate(x_batches):
               
                with torch.autocast(device_type='cuda', dtype=torch.float32, enabled=USE_AMP):
                    pred = model(batch) # stress rate
            
                    test_loss = l_fn(pred, y_batches[i])
            
                running_loss += test_loss.item() * batch.size()[0]
                total_samples += batch.size()[0]

            print('\r> Test: %d/%d' % (batch_idx + 1, num_batches), end='')

        test_loss = running_loss / total_samples

    return test_loss  

if __name__ == '__main__':
    
    # Loading configuration
    CONFIG_PATH = './config/'

    config = load_config(CONFIG_PATH, 'config_direct.yaml')

# ---------------------------------------------------------------
#                     Pytorch Configurations
# ---------------------------------------------------------------
    
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

        # Starting WANDB logging
        WANDB_RUN = wandb.init(project=config.wandb.project, 
                               entity=config.wandb.entity, 
                               config=WANDB_CONFIG)
        
        # Tagging the model
        MODEL_TAG = WANDB_RUN.name

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
        
    # Cleaning GPU cache
    torch.cuda.empty_cache() 

    # Splitting training dataset
    train_trials, test_trials = train_test_split(config.dirs.trials, config.data.split)

    if config.data.normalize.type != None:

        # Gathering statistics on training dataset
        stat_vars = get_data_stats(config.dirs.train, train_trials, ['tag'] + config.data.inputs, config.data.normalize.type)

        # Defining data transforms - normalization and noise addition 
        transform = transforms.Compose([
            get_data_transform(config.data.normalize.type, stat_vars)
        ])

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

    model = GRUModel(input_dim=len(config.data.inputs),
                     output_dim=len(config.data.outputs), 
                     hidden_dim=config.model.hidden_size, 
                     layer_dim=config.model.num_layers)

    model.apply(init_weights)
    model.to(DEVICE)

    l_fn = torch.nn.MSELoss()

    optimizer = torch.optim.AdamW(params=model.parameters(),
                                  weight_decay=config.train.w_decay, 
                                  lr=config.train.l_rate.max)
       
    steps_warmup = config.train.l_rate.warmup_steps
    
    steps_annealing = (config.train.epochs - (steps_warmup // (len(train_dataloader)*config.train.batch_divider // ITERS_TO_ACCUMULATE))) * (len(train_dataloader) * config.train.batch_divider // ITERS_TO_ACCUMULATE)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                           T_max=steps_annealing, 
                                                           eta_min=config.train.l_rate.min)
    
    warmup_lr = GradualWarmupScheduler(optimizer, 
                                       multiplier=1, 
                                       total_epoch=steps_warmup, 
                                       after_scheduler=scheduler)
    
    if config.train.use_amp:
        # Creates a GradScaler
        scaler = torch.cuda.amp.GradScaler()
    
    # Initializing the early_stopping object
    early_stopping = EarlyStopping(patience=config.train.e_stop.patience, 
                                   path=config.train.e_stop.checkpoint, 
                                   delta=config.train.e_stop.delta, 
                                   verbose=True)

    if config.wandb.log:
        wandb.watch(model, log='all')

    # Container variables for history purposes
    log_dict = {
        'epoch': {},
        't_loss': {},
        'v_loss': {},
        'clausius': {},
        'consist': {}
    }
    
    for t in range(config.train.epochs):

        print('\r--------------------\nEpoch [%d/%d]\n' % (t + 1, config.train.epochs))

        log_dict['epoch'][t] = t + 1

        # Train loop
        start_train = time.time()

        #--------------------------------------------------------------
        logs = train_loop(train_dataloader, model, l_fn, optimizer)
        #--------------------------------------------------------------

        log_dict['t_loss'][t] = logs['loss'] 
        log_dict['clausius'][t] = logs['clausius']
        log_dict['consist'][t] = logs['consist']
    
        end_train = time.time()

        print('. t_loss: %.6e | clausius: %.6e | consist: %.6e -> lr: %.4e -- %.3fs \n' % (log_dict['t_loss'][t], log_dict['clausius'][t], log_dict['consist'][t], warmup_lr._last_lr[0], end_train - start_train))

        # Test loop
        start_test = time.time()

        #-----------------------------------------------------------------------------------
        v_loss = test_loop(test_dataloader, model, l_fn)
        #-----------------------------------------------------------------------------------

        log_dict['v_loss'][t] = v_loss

        end_test = time.time()

        print('. v_loss: %.6e -- %.3fs' % (log_dict['v_loss'][t], end_test - start_test))

        if t > config.train.e_stop.start_at:
            early_stopping(log_dict['v_loss'][t], model)

        if config.wandb.log:
            wandb.log({
                'epoch': t,
                'l_rate': warmup_lr._last_lr[0],
                'train_loss': log_dict['t_loss'][t],
                'test_loss': log_dict['v_loss'][t],
                'clausius': log_dict['clausius'][t],
                'consist': log_dict['consist'][t]
            })

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
    

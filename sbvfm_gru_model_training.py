# ---------------------------------
#    Library imports
# ---------------------------------
import enum
import glob
import os
from re import X
import shutil
import joblib
import math
import pandas as pd
import random
import numpy as np
import math
import torch
import time
import wandb
from torch.autograd.functional import jacobian
import time
import random
import gc
import pyarrow.parquet as pq
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
from warmup_scheduler import GradualWarmupScheduler

from functions import (
    
    CoVWeightingLoss,
    GRUModel,
    EarlyStopping    

    )

from mesh_utils import (
    
    Element,
    get_b_inv,
    get_geom_limits,
    get_glob_strain_disp,
    get_b_bar,
    global_dof,
    read_mesh,

)

from vfm import (
    external_vw,
    get_ud_vfs,
    internal_vw
)

from vfm_loss import (
    
    SBVFLoss, 
    UDVFLoss
    
)

from constants import (
    FORMAT_PBAR
)


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
    def __init__(self, trials, root_dir, data_dir, features, outputs, info, centroids, scaler_x=None, transform=None, seq_len=1):
        self.seq_len = seq_len
        self.trials = trials
        self.root_dir = root_dir
        self.data_dir = data_dir
        self.scaler_x = scaler_x
        self.features = features
        self.outputs = outputs
        self.info = info
        self.centroids = centroids
        self.transform = transform
        self.files = [os.path.join(self.root_dir, self.data_dir , trial + '.parquet') for trial in self.trials]
        self.data = [pq.ParquetDataset(file).read_pandas(columns=self.features+self.outputs+self.info+['id','area']).to_pandas() for file in self.files]

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, idx):
        
        x = torch.from_numpy(self.data[idx][self.features].dropna().values).float()
        y = torch.from_numpy(self.data[idx][self.outputs].dropna().values).float()
        f = torch.from_numpy(self.data[idx][['fxx_t','fyy_t']].dropna().values).float()
        a = torch.from_numpy(self.data[idx]['area'][self.data[idx]['t']==0].dropna().values).float()
        t = torch.from_numpy(self.data[idx]['t'].values).float()
        
        t_pts = len(list(set(t.numpy())))
        n_elems = len(set(self.data[idx]['id'].values))

        # Adding a padding of zeros to the input data in order to make predictions start at zero
        pad_zeros = torch.zeros(self.seq_len * n_elems, x.shape[-1])
        
        x = torch.cat([pad_zeros, x], 0)

        if self.transform != None:
            
            x = self.transform(x)

        x = self.rolling_window(x.reshape(t_pts + self.seq_len,n_elems,-1), seq_size=self.seq_len)[:,:-1]
        #x = x.reshape(-1,*x.shape[2:])
        #t = self.rolling_window(t.reshape(t_pts,n_elems,-1), seq_size=self.seq_len)

        y = y.reshape(t_pts,n_elems,-1).permute(1,0,2)
        #y = y.reshape(-1,y.shape[-1])

        f = torch.mean(f.reshape(t_pts,n_elems,-1),1)
        a = a.unsqueeze(-1)
        #idx_ = torch.randperm(x.shape[0])

        #return x[idx_], y[idx_], f, t, t_pts, n_elems
        return x, y, f, t, a, self.centroids, t_pts, n_elems        
    
    def rolling_window(self, x, seq_size, step_size=1):
        # unfold dimension to make our rolling window
        return x.unfold(0,seq_size,step_size).permute(1,0,3,2)

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
    f_sum = lambda x: torch.sum(f(x), axis=0)
    return jacobian(f_sum, x,create_graph=True).permute(1,0,2)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train():

    # def get_grad_norm(model):
    #     total_norm = 0
    #     for p in model.parameters():
    #         if p.grad is not None:
    #             param_norm = p.grad.data.norm(2)
    #             total_norm += param_norm.item() ** 2
    #     total_norm = total_norm ** (1. / 2)
    #     return total_norm

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

    def train_loop(dataloader, model, l_fn, optimizer):
        '''
        Custom loop for neural network training, using mini-batches

        '''

        data_iter = iter(dataloader)

        num_batches = len(dataloader)
        
        losses = {}
        # f_loss = []
        # triax_loss = []
        # l_loss = []
               
        model.train()
        #l_fn.to_train()
        
        for batch_idx in range(len(dataloader)):

            # Extracting variables for training
            X_train, y_train, f, _, area, centroids, t_pts, n_elems = data_iter.__next__()
            
            X_train = X_train.squeeze(0)
            y_train = y_train.squeeze(0)
            centroids = centroids.squeeze(0)
            area = area.squeeze(0).to(DEVICE)
            f = f.squeeze(0).to(DEVICE)

            x_batches = torch.split(X_train, t_pts//BATCH_DIVIDER, 1)
            y_batches = torch.split(y_train, t_pts//BATCH_DIVIDER, 1)
            f_batches = torch.split(f, t_pts//BATCH_DIVIDER, 0)

            for i, batch in enumerate(x_batches):
                x = batch.reshape([-1,*batch.shape[-2:]])
                y = y_batches[i]
                f_ = f_batches[i]
                model.init_hidden(x.size(0), DEVICE)
                with torch.autocast(device_type='cuda', dtype=torch.float32, enabled=USE_AMP):
                    pred = model(x.to(DEVICE)) # stress

                #*******************************
                # ADD OTHER CALCULATIONS HERE!
                #*******************************
                v_strain =  V_STRAIN.unsqueeze(2).repeat(1,1,batch.size(1),1)
                v_disp = V_DISP.unsqueeze(1)
                a_ = area.unsqueeze(2).repeat(1,batch.size(1),1)

                w_int_ann = internal_vw(pred.reshape_as(y), v_strain, a_)
                #w_int_ann = torch.sum(torch.sum(pred * (torch.from_numpy(V_STRAIN) * area).to(DEVICE), -1, keepdim=True), 1)

                w_int = internal_vw(y.to(DEVICE), v_strain, a_)
                #w_int = torch.sum(torch.sum((batch_y * torch.from_numpy(V_STRAIN) * area).to(DEVICE), -1, keepdim=True), 1)

                #w_ext = torch.sum((f[t] * V_DISP).to(DEVICE), -1, keepdim=True)
                w_ext = external_vw(f_, v_disp)


                with torch.autocast(device_type='cuda', dtype=torch.float32, enabled=USE_AMP):
                    
                    #l_tuples = [(pred[:,0], y_batches[i][:,0].to(device)), (pred[:,1], y_batches[i][:,1].to(device)), (pred[:,1], y_batches[i][:,1].to(device))]
                    
                    loss = l_fn(w_int_ann, w_ext)
                
                    #loss = l_fn(l_tuples)
                
                scaler.scale(loss/ITERS_TO_ACCUMULATE).backward()
            
                if ((t + 1) % ITERS_TO_ACCUMULATE == 0) or (t + 1 == t_pts):    
                    
                    scaler.step(optimizer)
                    scaler.update()
                    
                    warmup_lr.step()

                    optimizer.zero_grad(set_to_none=True)

                # Saving loss values
                losses[t] = loss.item()
                # f_loss.append(0.0)
                # triax_loss.append(0.0)
                # l_loss.append(0.0  )

            print('\r>Train: %d/%d' % (batch_idx + 1, num_batches), end='')
              
        #-----------------------------
        return np.fromiter(losses.values(), dtype=np.float32)

    def test_loop(dataloader, model, l_fn):

        data_iter = iter(dataloader)

        num_batches = len(dataloader)
        
        test_losses = []

        model.eval()
        #l_fn.to_eval()

        with torch.no_grad():

            for batch_idx in range(len(dataloader)):
            
                X_test, _, f, _, area, centroids, t_pts, n_elems = data_iter.__next__()
            
                X_test = X_test.squeeze(0)
                # y_test = y_test.squeeze(0)
                centroids = centroids.squeeze(0)
                area = area.squeeze(0).to(DEVICE)
                f = f.squeeze(0).to(DEVICE)

                for t in range(t_pts):
                
                    batch_x = X_test[:,t]
                    #batch_y = y_test[:,t]

                    model.init_hidden(batch_x.size(0), DEVICE)
                    with torch.autocast(device_type='cuda', dtype=torch.float32, enabled=USE_AMP):
                        pred = model(batch_x.to(DEVICE)) # stress
        
                    w_int_ann = internal_vw(pred, V_STRAIN, area)
                    
                    w_ext = external_vw(f[t], V_DISP)
                
                    with torch.autocast(device_type='cuda', dtype=torch.float32, enabled=USE_AMP):
                        
                        #l_tuples = [(pred[:,0], y_test[:,0]), (pred[:,1], y_test[:,1]), (l_triax,)]
                    
                        #test_loss = l_fn(l_tuples)
                        
                        test_loss =  l_fn(w_int_ann, w_ext)
                
                    test_losses.append(test_loss.item())

                print('\r>Test: %d/%d' % (batch_idx + 1, num_batches), end='')

        return np.array(test_losses)

#////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# 
#                               NEURAL NETWORK TRAINING SCRIPT - Configuration
# 
#////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#       
#-------------------------------------------------------------------------------------------------------------------
#                                                   PYTORCH SETUP
#-------------------------------------------------------------------------------------------------------------------

    # Disabling Debug APIs
    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)
    torch.autograd.profiler.emit_nvtx(False)

    # Default floating point precision
    torch.set_default_dtype(torch.float32)

    # Cleaning GPU cache
    torch.cuda.empty_cache()

    # Defining current device
    if torch.cuda.is_available():  
        dev = "cuda:0"
        KWARGS = {'num_workers': 0, 'pin_memory': False} 
    else:  
        dev = "cpu"  
        KWARGS = {'num_workers': 0}
    
    DEVICE = torch.device(dev)

#-------------------------------------------------------------------------------------------------------------------
#                                                  RANDOM SEED SETUP
#-------------------------------------------------------------------------------------------------------------------    
    
    # Global seed
    SEED = 9567

    os.environ['PYTHONHASHSEED']=str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

#-------------------------------------------------------------------------------------------------------------------
#                                                   PROGRAM MODE
#-------------------------------------------------------------------------------------------------------------------

    # TO BE ADDED

#-------------------------------------------------------------------------------------------------------------------
#                                               DATASET CONFIGURATIONS
#-------------------------------------------------------------------------------------------------------------------
    
    # Data directories
    TRAIN_DIR = 'data/training_multi/crux-plastic/'

    # Dataset split
    TEST_SIZE = 0.3

    # Defining variables of interest
    FEATURES = ['exx_t','eyy_t','exy_t']
    OUTPUTS = ['sxx_t','syy_t','sxy_t']
    INFO = ['tag','inc','t','cent_x','cent_y','fxx_t','fyy_t']

#-------------------------------------------------------------------------------------------------------------------
#                                              MODEL HYPERPARAMETERS
#-------------------------------------------------------------------------------------------------------------------

    # Sequence length
    SEQ_LEN = 4

    # No. inputs/outputs
    N_INPUTS = len(FEATURES)
    N_OUTPUTS = len(OUTPUTS)
    
    # No. hidden neurons - size of list indicates no. of hidden layers
    HIDDEN_UNITS = [32]

    # Stacked GRU units
    GRU_LAYERS = 2

#-------------------------------------------------------------------------------------------------------------------
#                                               TRAINING SETTINGS
#-------------------------------------------------------------------------------------------------------------------

    # Learning rate
    L_RATE = 0.005

    # Weight decay
    L2_REG = 0.001

    # No. epochs
    EPOCHS = 10000

    # No. of warmup steps for cosine annealing scheduler
    WARM_STEPS = 1920 # 5 epochs

    # No. of epochs to trigger early-stopping
    ES_PATIENCE = 500

    # No. of after which to start early-stopping check
    ES_START = 200

    # No. of batches to divide training data into (acts upon time points)
    BATCH_DIVIDER = 16

    # No. of steps to acumulate gradients
    ITERS_TO_ACCUMULATE = 1

    # Automatic mixed precision
    USE_AMP = True

#-------------------------------------------------------------------------------------------------------------------
#                                               VFM CONFIGURATION
#-------------------------------------------------------------------------------------------------------------------

    # Selecting VFM formulation:
    #   ud - user-defined vf's
    #   sb - sensitivity-based
    VFM_TYPE = 'ud'

#-------------------------------------------------------------------------------------------------------------------
#                                           MESH INFORMATION FOR VFM
#-------------------------------------------------------------------------------------------------------------------

    # Reading mesh file
    MESH, CONNECTIVITY, DOF = read_mesh(TRAIN_DIR)

    # Defining geometry limits
    X_MIN, X_MAX, Y_MIN, Y_MAX = get_geom_limits(MESH)

    # Element centroids
    CENTROIDS = pd.read_csv(os.path.join(TRAIN_DIR,'centroids.csv'), usecols=['cent_x','cent_y']).values

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

    if VFM_TYPE == 'sb':  # Sensitivity-based VFM

        # Total degrees of freedom
        TOTAL_DOF = MESH.shape[0] * 2

        # Global degrees of freedom
        GLOBAL_DOF = list(range(TOTAL_DOF)) 

        # Constructing element properties based on mesh info
        ELEMENTS = [Element(CONNECTIVITY[i,:], MESH[CONNECTIVITY[i,1:]-1,1:], DOF[i,:]) for i in range(CONNECTIVITY.shape[0])]

        # Global strain-displacement matrix
        B_GLOB = get_glob_strain_disp(ELEMENTS, TOTAL_DOF, BC_SETTINGS)

        # Modified strain-displacement matrix and active defrees of freedom
        B_BAR, ACTIVE_DOF = get_b_bar(BC_SETTINGS, B_GLOB, GLOBAL_DOF)

        # Inverse of modified strain-displacement matrix
        B_INV = get_b_inv(B_BAR)
    
    elif VFM_TYPE == 'ud':  # User-defined VFM

        # Maximum dimensions of specimen
        WIDTH = X_MAX - X_MIN
        HEIGHT = Y_MAX - Y_MIN

        # Coordinates for traction surfaces
        SURF_COORDS = [X_MAX, Y_MAX]

        # Computing virtual fields
        TOTAL_VFS, V_DISP, V_STRAIN = get_ud_vfs(CENTROIDS, SURF_COORDS, WIDTH, HEIGHT)

        V_DISP = torch.from_numpy(V_DISP).to(DEVICE)
        V_STRAIN = torch.from_numpy(V_STRAIN).to(DEVICE)

#-------------------------------------------------------------------------------------------------------------------
#                                               WANDB CONFIGURATIONS
#-------------------------------------------------------------------------------------------------------------------
    
    # Project name
    PROJ = 'sbvfm_indirect_crux_gru'
    
    # WANDB logging
    WANDB_LOG = False

    if WANDB_LOG:

        # WANDB model details
        WANDB_CONFIG = {
            'inputs': N_INPUTS,
            'outputs': N_OUTPUTS,
            'hidden_layers': len(HIDDEN_UNITS),
            'hidden_units': f'{*HIDDEN_UNITS,}',
            'stack_units': GRU_LAYERS,        
            'epochs': EPOCHS,
            'l_rate': L_RATE,
            'l2_reg': L2_REG
        }

        # Starting WANDB logging
        WANDB_RUN = wandb.init(project=PROJ, entity="rmbl", config=WANDB_CONFIG)
        
        # Tagging the model
        MODEL_TAG = WANDB_RUN.name

    else:

        # Tagging the model
        MODEL_TAG = time.strftime("%Y%m%d-%H%M%S")

#-------------------------------------------------------------------------------------------------------------------
#                                               OUTPUT CONFIGURATION
#-------------------------------------------------------------------------------------------------------------------

    # Model name
    MODEL_NAME = f"{MODEL_TAG}-[{N_INPUTS}-GRUx{GRU_LAYERS}-{*HIDDEN_UNITS,}-{N_OUTPUTS}]-{TRAIN_DIR.split('/')[-2]}"
    
    # Temp folder for model checkpoint
    TEMP_DIR = os.path.join('temp', MODEL_TAG)

    # Model checkpoint path
    CHECKPOINT_DIR = os.path.join(TEMP_DIR, 'checkpoint.pt')
    
    # Output paths
    DIR_TASK = os.path.join('outputs', PROJ)
    DIR_LOSS = os.path.join(DIR_TASK,'loss')
    DIR_STATS = os.path.join(DIR_TASK, 'stats')
    DIR_MODELS = os.path.join(DIR_TASK, 'models')
    DIR_VAL = os.path.join(DIR_TASK, 'val')
    DIR_LOGS = os.path.join(DIR_TASK, 'logs')

    SCALER_FILE = os.path.join(DIR_MODELS, MODEL_NAME + '-scaler_x.pkl')
    MODEL_FILE = os.path.join(DIR_MODELS, MODEL_NAME,'.pt')
    ARCH_FILE = os.path.join(DIR_MODELS, MODEL_TAG, '-arch.pkl')

#-------------------------------------------------------------------------------------------------------------------
#                                           CREATING OUTPUT DIRECTORIES
#-------------------------------------------------------------------------------------------------------------------

    # Creating output directories
    directories = [TEMP_DIR, DIR_TASK, DIR_LOSS, DIR_STATS, DIR_MODELS, DIR_VAL, DIR_LOGS]

    for dir in directories:
        try:
            os.makedirs(dir)
        except FileExistsError:
            pass

#-------------------------------------------------------------------------------------------------------------------
#                                                   LOSS FUNCTIONS
#-------------------------------------------------------------------------------------------------------------------

    # Loss functions for VFM
    if VFM_TYPE == 'sb':
    
        l_fn = SBVFLoss()
    
    elif VFM_TYPE == 'ud':

        l_fn = UDVFLoss(normalize='wint')

    # Initializing GradScaler object
    scaler = torch.cuda.amp.GradScaler()

#////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# 
#                                           NEURAL NETWORK TRAINING SCRIPT
# 
#////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    # Loading mechanical trial tags to separate train/test datasets
    trials = pd.read_csv(os.path.join(TRAIN_DIR,'t_trials.csv'), index_col=False, header=0)
    trials = list(trials['0'])
    trials = random.sample(trials,len(trials))

    test_trials = random.sample(trials, math.ceil(len(trials)*TEST_SIZE))
    train_trials = list(set(trials).difference(test_trials))    

    # Gathering statistics on training dataset
    dir = os.path.join(TRAIN_DIR,'processed/')
    file_list = glob.glob(os.path.join(dir, f'*.parquet'))
    file_list = [file for file in file_list if file.split('\\')[-1].split('.')[0] in train_trials]
    
    df_list = [pq.ParquetDataset(file).read_pandas(columns=FEATURES).to_pandas() for file in tqdm(file_list,desc='Importing training data', bar_format=FORMAT_PBAR)]

    raw_data = pd.concat(df_list)

    min_ = torch.min(torch.from_numpy(raw_data.values).float(),0).values
    max_ = torch.max(torch.from_numpy(raw_data.values).float(),0).values
    #std, mean = torch.std_mean(torch.from_numpy(input_data.values),0)

    # Cleaning workspace from useless variables
    del df_list
    del file_list
    gc.collect()

    # Defining data transforms
    transform = transforms.Compose([
        MinMaxScaler(min_,max_),
        #Normalize(mean.tolist(), std.tolist()),
        #transforms.RandomApply([AddGaussianNoise(0., 1.)],p=0.15)
    ])

    # Preparing dataloaders for mini-batch training
    train_dataset = CruciformDataset(train_trials, TRAIN_DIR, 'processed', FEATURES, OUTPUTS, INFO, CENTROIDS, transform=transform, seq_len=SEQ_LEN)
    test_dataset = CruciformDataset(test_trials, TRAIN_DIR, 'processed', FEATURES, OUTPUTS, INFO, CENTROIDS, transform=transform, seq_len=SEQ_LEN)
    
    train_dataloader = DataLoader(train_dataset, shuffle=True, **KWARGS)
    test_dataloader = DataLoader(test_dataset, **KWARGS)
    
    # Initializing neural network model
    model = GRUModel(input_dim=N_INPUTS, hidden_dim=HIDDEN_UNITS, layer_dim=GRU_LAYERS, output_dim=N_OUTPUTS)

    # Initializing dense layer weights
    model.apply(init_weights)
    
    # Transfering model to current device
    model.to(DEVICE)

    # Training variables
    #epochs = 10000

    # Optimization variables
    #learning_rate = 0.005
    #lr_mult = 1.0

    #params = layer_wise_lr(model, lr_mult=lr_mult, learning_rate=learning_rate)
    
    #l_fn = CoVWeightingLoss(device=device, n_losses=3)
    #l_fn = torch.nn.MSELoss()

    #weight_decay = 0.001

    # Initializing optimizer
    optimizer = torch.optim.AdamW(params=model.parameters(), weight_decay=L2_REG, lr=L_RATE)
    
    # Defining cosine annealing steps
    steps_annealing = (EPOCHS - (WARM_STEPS // (len(train_dataloader)*BATCH_DIVIDER // ITERS_TO_ACCUMULATE))) * (len(train_dataloader)*BATCH_DIVIDER // ITERS_TO_ACCUMULATE)

    # Initializing cosine annealing with warmup lr scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps_annealing, eta_min=1e-3)
    warmup_lr = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=WARM_STEPS, after_scheduler=scheduler)  

    # f_loss = []
    # l_loss = []
    # triax_loss = []

    # alpha1 = []
    # alpha2 = []
    # alpha3 = []
    # alpha4 = []

    # Initializing the early_stopping object
    early_stopping = EarlyStopping(patience=ES_PATIENCE, path=CHECKPOINT_DIR, verbose=True)

    # Preparing Weights & Biases logging
    if WANDB_LOG:
        
        wandb.watch(model, log='all')

    # Dictionary to log training variables
    log_dict = {
        'epoch': {},
        'loss': {
            'train': {},
            'test': {},
        },
        'constraints': {},
    }

    # Training loop
    for t in range(EPOCHS):

        print('\r--------------------\nEpoch [%d/%d]' % (t + 1, EPOCHS))

        log_dict['epoch'][t] = t + 1

        #--------------------------------------------------------------
        #                       TRAIN LOOP
        #--------------------------------------------------------------
        start_t = time.time()

        batch_losses = train_loop(train_dataloader, model, l_fn, optimizer)
    
        log_dict['loss']['train'][t] = np.mean(batch_losses)
    
        end_t = time.time()
        delta_t = end_t - start_t
        #--------------------------------------------------------------

        # Printing loss to console
        try:
            print('. t_loss: %.6e -> lr: %.4e -- %.3fs \n' % (log_dict['loss']['train'][t], warmup_lr._last_lr[0], delta_t))
        except:
            print('. t_loss: %.6e -- %.3fs' % (log_dict['loss']['train'][t], delta_t))

        #--------------------------------------------------------------
        #                       TEST LOOP
        #--------------------------------------------------------------
        start_t = time.time()

        batch_val_losses = test_loop(test_dataloader, model, l_fn)
        
        log_dict['loss']['test'][t] = np.mean(batch_val_losses)

        end_t = time.time()
        delta_t = end_t - start_t

        #--------------------------------------------------------------

        # Printing loss to console
        print('. v_loss: %.6e -- %.3fs' % (log_dict['loss']['test'][t], delta_t))

        # Check early-stopping
        if t > ES_START:
            early_stopping(log_dict['loss']['test'][t], model)

        if WANDB_LOG:
            wandb.log({
                'epoch': t,
                'l_rate': warmup_lr._last_lr[0],
                'train_loss': log_dict['loss']['train'][t],
                'test_loss': log_dict['loss']['test'][t],
                # 'f_loss': f_loss[t],
                # 't_loss': triax_loss[t],
                # 'mse_loss': l_loss[t],
                # 'alpha_1': alpha1[t],
                # 'alpha_2': alpha2[t],
                # 'alpha_3': alpha3[t],
                # 'alpha_4': alpha4[t]
            })

        # Triggering early-stopping
        if early_stopping.early_stop:
            print("Early stopping")
            break
    
    print("Done!")

    # Load the checkpoint with the best model
    model.load_state_dict(torch.load(CHECKPOINT_DIR))

    epoch = np.fromiter(log_dict['epoch'].values(), dtype=np.int16).reshape(-1,1)
    train_loss = np.fromiter(log_dict['loss']['train'].values(), dtype=np.float32).reshape(-1,1)
    val_loss = np.fromiter(log_dict['loss']['test'].values(), dtype=np.float32).reshape(-1,1)

    history = pd.DataFrame(np.concatenate([epoch, train_loss, val_loss], axis=1), columns=['epoch','loss','val_loss'])
    
    history.to_csv(os.path.join(DIR_LOSS, MODEL_NAME, '.csv'), sep=',', encoding='utf-8', header='true')

    #plot_history(history, output_loss, True, task)

    torch.save(model.state_dict(), os.path.join(DIR_MODELS, MODEL_NAME, '.pt'))
    
    joblib.dump([min_, max_], os.path.join(DIR_MODELS, MODEL_NAME + '-scaler_x.pkl'))
    joblib.dump([FEATURES, OUTPUTS, INFO, HIDDEN_UNITS, GRU_LAYERS, SEQ_LEN], os.path.join(DIR_MODELS, MODEL_TAG, '-arch.pkl'))

    # At the end of training, save the model artifact
    if WANDB_LOG:
        
        # Name this artifact after the current run
        model_artifact_name = WANDB_RUN.id + '_' + MODEL_NAME
        # Create a new artifact
        model = wandb.Artifact(model_artifact_name, type='model')
        # Add files to the artifact
        model.add_file(MODEL_FILE)
        model.add_file(SCALER_FILE)
        model.add_file(ARCH_FILE)
        # Log the model to W&B
        WANDB_RUN.log_artifact(model)
        # Call finish if you're in a notebook, to mark the run as done
        WANDB_RUN.finish()

    # Deleting temp folder
    try:
        shutil.rmtree(TEMP_DIR)
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

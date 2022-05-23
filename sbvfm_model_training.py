# ---------------------------------
#    Library and function imports
# ---------------------------------
import os
import shutil
import joblib
from constants import *
from functions import (
    ICNN,
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
    EarlyStopping,
    NeuralNetwork,
    Element)
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

from tqdm import tqdm

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
    def __init__(self, deformation, stress, force, coord, info, list_IDs, batch_size, shuffle=False, std=True, t_pts=1):
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
                X, y, f, coord, tag, t = self.__data_generation(indexes)
            else:
                X, y, f, coord, tag, t = self.__data_generation(np.random.permutation(indexes))
        else:
            X, y, f, coord, tag, t = self.__data_generation(indexes)
        
        return X, y, f, coord, tag, t

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
        X = np.asarray(self.X.iloc[list_IDs_temp], dtype=np.float64)
        y = np.asarray(self.y.iloc[list_IDs_temp], dtype=np.float64)
        f = np.asarray(self.f.iloc[list_IDs_temp], dtype=np.float64)
        coord = np.asarray(self.coord.iloc[list_IDs_temp], dtype=np.float64)
        tag = self.tag.iloc[list_IDs_temp]
        t = self.t.iloc[list_IDs_temp]
        return X, y, f, coord, tag, t

# -------------------------------
#       Method definitions
# -------------------------------

def batch_jacobian(f, x):
    f_sum = lambda x: torch.sum(f(x), axis=0)
    return jacobian(f_sum, x,create_graph=True).permute(1,0,2)


def sigma_deltas(model, X_train, pred, param_dicts):
    
    delta_stress = torch.zeros((len(param_dicts), pred.shape[0], pred.shape[1]))
    
    model.eval()
    with torch.no_grad():
        for i,param_dict in enumerate(param_dicts):
            model.load_state_dict(param_dict)
            pred_eval = model(X_train)
            delta_stress[i,:,:] = (pred - pred_eval)

    return delta_stress

def sbv_fields(d_sigma, b_glob, b_inv, n_elems, bcs, active_dof):

    n_dof = b_glob.shape[-1]

    # Reshaping d_sigma for least-square system
    d_s = torch.reshape(d_sigma,(d_sigma.shape[0], T_PTS, n_elems * d_sigma.shape[-1],1))

    v_u = torch.zeros((d_s.shape[0],d_s.shape[1], n_dof, 1))

    # Computing virtual displacements (all dofs)
    v_u[:, :, active_dof] = b_inv @ d_s
    
    # Prescribing displacements
    v_u, v_disp = prescribe_u(v_u, bcs)

    v_strain = torch.reshape(b_glob @ v_u, d_sigma.shape)
    
    return v_u, v_disp, v_strain

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
   

def train_loop(dataloader, model, loss_fn, optimizer):
    '''
    Custom loop for neural network training, using mini-batches

    '''
    global w_virt

    mdl = copy.deepcopy(model)
    param_dicts = param_deltas(mdl)
    n_vfs = len(param_dicts)

    num_batches = len(dataloader)
    losses = torch.zeros(num_batches)
    v_work_real = torch.zeros(num_batches)
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
         
        # Extracting element ids
        id = coord_torch[:,1]
        # Reshaping and sorting element ids array
        # id_reshaped = torch.reshape(id, (t_pts,n_elems,1))
        # indices = id_reshaped[:, :, 0].sort()[1]
        
        # Extracting element area
        area = torch.reshape(coord_torch[:,4],[batch_size,1])

        # Computing model stress prediction
        pred=model(X_train)

        # Reshaping prediction to sort according to the sorted element ids
        # pred_reshaped = torch.reshape(pred,(t_pts,n_elems,pred.shape[-1]))
        # pred_ = pred_reshaped[torch.arange(pred_reshaped.size(0)).unsqueeze(1), indices]

        # y_reshaped = torch.reshape(y_train,((t_pts,n_elems,y_train.shape[-1])))
        # y_ = y_reshaped[torch.arange(y_reshaped.size(0)).unsqueeze(1), indices]
        # y_ = torch.reshape(y_,y_train.shape)

        # # Reshaping input tensor to sort according to the sorted element ids
        # X_train_reshaped = torch.reshape(X_train,(t_pts,n_elems,X_train.shape[-1])).detach()
        # X_train_sorted = X_train_reshaped[torch.arange(X_train_reshaped.size(0)).unsqueeze(1), indices].detach()

        # Reshaping tensors back to original shape in order to be able to pass through ANN model
        # pred_ = torch.reshape(pred_,pred.shape)
        # X_train_sorted = torch.reshape(X_train_sorted,X_train.shape)

        # Computing stress perturbation from perturbed model parameters
        d_sigma = sigma_deltas(copy.deepcopy(model), X_train.detach(), pred.detach(), param_dicts)

        # Computing sensitivity-based virtual fields
        v_u, v_disp, v_strain = sbv_fields(d_sigma, b_glob, b_inv, n_elems, bcs, active_dof)

        # Computing predicted virtual work       
        int_work = torch.sum(torch.sum(torch.reshape((pred * v_strain * area * ELEM_THICK),[n_vfs,t_pts,n_elems,3]),-1),-1,keepdim=True)
       
        # Computing real virtual work
        int_work_real = torch.sum(torch.sum(torch.reshape((y_train * v_strain * area * ELEM_THICK),[n_vfs,t_pts,n_elems,3]),-1),-1,keepdim=True)

        f = f_train[:,:2][::n_elems,:]

        # Computing external virtual work
        ext_work = torch.sum(torch.reshape(f,[t_pts,1,2])*v_disp,-1)

        # Computing losses        
        loss = loss_fn(int_work,ext_work)
        cost = loss_fn(int_work_real, ext_work)

        # Backpropagation and weight's update
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        # Saving loss values
        losses[batch] = loss
        v_work_real[batch] = cost

        tags = tag[::n_elems].values.tolist()
        incs = inc[::n_elems].values.tolist()
        
        for i,(n,j) in enumerate(tuple(zip(tags,incs))):
            VFs[n]['u'][j] = v_u[:,i].detach().numpy()
            VFs[n]['e'][j] = torch.reshape(v_strain,[n_vfs,t_pts,n_elems,3])[:,i].detach().numpy()
            w_virt[n]['w_int'][j] = int_work[:,i].detach().numpy()
            w_virt[n]['w_int_real'][j] = int_work_real[:,i].detach().numpy()
            w_virt[n]['w_ext'][j] = ext_work[:,i].detach().numpy()

        print('\r>Train: %d/%d' % (batch + 1, num_batches), end='')
    
    return losses, v_work_real, v_disp, v_strain, v_u
    
def test_loop(dataloader, model, loss_fn, v_disp, v_strain):
    
    # param_dicts = param_deltas(copy.deepcopy(model))
    # n_vfs = len(param_dicts)

    global w_virt

    num_batches = len(dataloader)
    test_losses = torch.zeros(num_batches)
   
    t_pts = dataloader.t_pts
    n_elems = batch_size//t_pts
    n_vfs = v_strain.shape[0]

    model.eval()
    with torch.no_grad():

        for batch in range(num_batches):

            # Extracting variables for testing
            X_test, y_test, f_test, coord, tag, inc = dataloader[batch]
            
            # Converting to pytorch tensors
            if train_generator.std == True:
                X_test = torch.from_numpy(train_generator.scaler_x.transform(X_test))
            else:
                X_test = torch.tensor(X_test, dtype=torch.float64)

            y_test = torch.from_numpy(y_test)
            f_test = torch.from_numpy(f_test) + 0.0
            coord_torch = torch.from_numpy(coord)
            
            # Extracting element ids
            id = coord_torch[:,1]

            area = torch.reshape(coord_torch[:,4],[batch_size,1])

            pred = model(X_test)

            # # Reshaping prediction to sort according to the sorted element ids
            # pred_reshaped = torch.reshape(pred,(t_pts,n_elems,pred.shape[-1]))
            # pred_ = pred_reshaped[torch.arange(pred_reshaped.size(0)).unsqueeze(1), indices]

            # y_reshaped = torch.reshape(y_test,((t_pts,n_elems,y_test.shape[-1])))
            # y_ = y_reshaped[torch.arange(y_reshaped.size(0)).unsqueeze(1), indices]
            # y_ = torch.reshape(y_,y_test.shape)

            # # Reshaping input tensor to sort according to the sorted element ids
            # X_test_reshaped = torch.reshape(X_test,(t_pts,n_elems,X_test.shape[-1])).detach()
            # X_test_sorted = X_test_reshaped[torch.arange(X_test_reshaped.size(0)).unsqueeze(1), indices].detach()

            # # Reshaping tensors back to original shape in order to be able to pass through ANN model
            # pred_ = torch.reshape(pred_,pred.shape)
            # X_test_sorted = torch.reshape(X_test_sorted,X_test.shape)

            # param_dicts = param_deltas(model)
            # n_vfs = len(param_dicts)

            # Computing stress perturbation from perturbed model parameters
            #d_sigma = sigma_deltas(model, X_test.detach(), pred.detach(), param_dicts)

            # Computing sensitivity-based virtual fields
            #v_u, v_disp, v_strain = sbv_fields(d_sigma, b_glob, b_inv, n_elems, bcs)

            int_work = torch.sum(torch.sum(torch.reshape((pred * v_strain * area * ELEM_THICK),[n_vfs,t_pts,n_elems,3]),-1),-1,keepdim=True)

            int_work_real = torch.sum(torch.sum(torch.reshape((y_test * v_strain * area * ELEM_THICK),[n_vfs,t_pts,n_elems,3]),-1),-1,keepdim=True)

            f = f_test[:,:2][::n_elems,:]
            
            ext_work = torch.sum(torch.reshape(f,[t_pts,1,2])*v_disp,-1)

            # Computing losses        
            # test_loss = loss_fn(int_work, ext_work)
            test_loss = loss_fn(int_work, ext_work)
            test_losses[batch] = test_loss

            tags = tag[::n_elems].values.tolist()
            incs = inc[::n_elems].values.tolist()
            
            for i,(n,j) in enumerate(tuple(zip(tags,incs))):
                VFs[n]['u'][j] = v_u[:,i].detach().numpy()
                VFs[n]['e'][j] = torch.reshape(v_strain,[n_vfs,t_pts,n_elems,3])[:,i].detach().numpy()
                w_virt[n]['w_int'][j] = int_work[:,i].detach().numpy()
                w_virt[n]['w_int_real'][j] = int_work_real[:,i].detach().numpy()
                w_virt[n]['w_ext'][j] = ext_work[:,i].detach().numpy()

            print('\r>Test: %d/%d' % (batch + 1, num_batches), end='')

    return test_losses

# -------------------------------
#           Main script
# -------------------------------

# Creating temporary folder
try:
    os.makedirs('./temp')          
except FileExistsError:
    pass

# Initiating wandb
#wandb.init(project="pytorch_linear_model", entity="rmbl")

torch.set_default_dtype(torch.float64)
# torch.set_printoptions(precision=8)

# Specifying random seed
random.seed(SEED)
torch.manual_seed(SEED)

mesh, connectivity, dof = read_mesh(TRAIN_MULTI_DIR)

# Defining geometric limits
x_min = min(mesh[:,1])
x_max = max(mesh[:,1])
y_min = min(mesh[:,-1])
y_max = max(mesh[:,-1])

# Total degrees of freedom
total_dof = mesh.shape[0] * 2

# Defining edge boundary conditions
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
elements = [Element(connectivity[i,:],mesh[connectivity[i,1:]-1,1:],dof[i,:]) for i in tqdm(range(connectivity.shape[0]))]

# Assembling global strain-displacement matrices
b_glob, b_bar, b_inv, active_dof = global_strain_disp(elements, total_dof, bcs)

# Loading data
df_list, _ = load_dataframes(TRAIN_MULTI_DIR)

# Merging training data
data = pd.concat(df_list, axis=0, ignore_index=True)

T_PTS = 51

# Performing test/train split
partition = {"train": None, "test": None}

if T_PTS==DATA_SAMPLES:
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
    test_trials = random.sample(trials, math.floor(len(trials)*TEST_SIZE))
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

    batch_size = len(data_by_batches[0]) * T_PTS

    data = pd.concat(data_by_batches).reset_index(drop=True)

    batches = np.array_split(data.index.tolist(),len(data_by_batches))
    idx = list(range(len(batches)))
    test_batch_idxs = random.sample(idx, math.floor(len(batches)*TEST_SIZE))
    test_batch_idxs.sort()
    train_batch_idxs = list(set(idx).difference(test_batch_idxs))

    partition['train'] = list(itertools.chain.from_iterable([batches[i].tolist() for i in train_batch_idxs]))
    partition['test'] = list(itertools.chain.from_iterable([batches[i].tolist() for i in test_batch_idxs]))

# Selecting model features
X, y, f, coord, info = select_features_multi(data)

# Preparing data generators for mini-batch training
train_generator = DataGenerator(X, y, f, coord, info, partition["train"], batch_size, False, std=True, t_pts=T_PTS)
test_generator = DataGenerator(X, y, f, coord, info, partition['test'], batch_size, False, std=False, t_pts=T_PTS)

# Model variables
N_INPUTS = X.shape[1]
N_OUTPUTS = y.shape[1]

N_UNITS = 4
H_LAYERS = 2

model_1 = NeuralNetwork(N_INPUTS, N_OUTPUTS, N_UNITS, H_LAYERS)

model_1.apply(init_weights)

# Training variables
epochs = 4000

# Optimization variables
learning_rate = 0.1
loss_fn = sbvf_loss
f_loss = torch.nn.MSELoss()

optimizer = torch.optim.Adam(params=list(model_1.parameters()), lr=learning_rate)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=30, factor=0.2, threshold=1e-3, min_lr=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=20, T_mult=1, eta_min=0.001)

#scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,0.99)

# Container variables for history purposes
train_loss = []
v_work = []
val_loss = []
epochs_ = []
# Initializing the early_stopping object
early_stopping = EarlyStopping(patience=75, delta=1e-7, path='temp/checkpoint.pt', verbose=True)

#wandb.watch(model_1)
VFs = {key: {k: dict.fromkeys(set(info['inc'])) for k in ['u','e']} for key in set(info['tag'])}
#VFs = {key: dict.fromkeys(set(info['inc'])) for key in set(info['tag'])}
w_virt = {key: {k: dict.fromkeys(set(info['inc'])) for k in ['w_int','w_int_real','w_ext']} for key in set(info['tag'])}

for t in range(epochs):

    print('\r--------------------\nEpoch [%d/%d]' % (t + 1, epochs))

    epochs_.append(t+1)
    
    start_epoch = time.time()

    #Shuffling batches
    train_generator.on_epoch_end()
    
    # Train loop
    start_train = time.time()
    batch_losses, batch_v_work, v_disp, v_strain, v_u = train_loop(train_generator, model_1, loss_fn, optimizer)

    train_loss.append(torch.mean(batch_losses).item())
    v_work.append(torch.mean(batch_v_work).item())

    end_train = time.time()
    
    #Apply learning rate scheduling if defined
    try:
        scheduler.step()
        #scheduler.step()
        print('. t_loss: %.3e -> lr: %.3e // [v_work] -> %.3e -- %.3fs' % (train_loss[t], scheduler._last_lr[0], v_work[t], end_train - start_train))
    except:
        print('. t_loss: %.3e // [v_work] -> %.3e -- %.3fs' % (train_loss[t], v_work[t], end_train - start_train))

    # Test loop
    start_test = time.time()

    batch_val_losses = test_loop(test_generator, model_1, loss_fn, v_disp, v_strain)

    val_loss.append(torch.mean(batch_val_losses).item())

    end_test = time.time()

    print('. v_loss: %.3e -- %.3fs' % (val_loss[t], end_test - start_test))

    end_epoch = time.time()

    if t > 50:
        # # Check validation loss for early stopping
        early_stopping(val_loss[t], model_1)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    # wandb.log({
    #     "Epoch": t,
    #     "Train Loss": train_loss[t],
    #     "Test Loss": val_loss[t],
    #     #"Learning Rate": scheduler._last_lr[0]
    #     }
    #     )

print("Done!")

# load the last checkpoint with the best model
model_1.load_state_dict(torch.load('temp/checkpoint.pt'))

epochs_ = np.reshape(np.array(epochs_), (len(epochs_),1))
train_loss = np.reshape(np.array(train_loss), (len(train_loss),1))
val_loss = np.reshape(np.array(val_loss), (len(val_loss),1))

history = pd.DataFrame(np.concatenate([epochs_, train_loss, val_loss], axis=1), columns=['epoch','loss','val_loss'])


task = r'[%i-%ix%i-%i]-%s-%i-VFs' % (N_INPUTS, N_UNITS, H_LAYERS, N_OUTPUTS, TRAIN_MULTI_DIR.split('/')[-2], v_strain.shape[0])

output_task = 'outputs/' + TRAIN_MULTI_DIR.split('/')[-2] + '_sbvf'
output_loss = 'outputs/' + TRAIN_MULTI_DIR.split('/')[-2] + '_sbvf/loss/'
output_stats = 'outputs/' + TRAIN_MULTI_DIR.split('/')[-2] + '_sbvf/stats/'
output_models = 'outputs/' + TRAIN_MULTI_DIR.split('/')[-2] + '_sbvf/models/'
output_val = 'outputs/' + TRAIN_MULTI_DIR.split('/')[-2] + '_sbvf/val/'
output_logs = 'outputs/' + TRAIN_MULTI_DIR.split('/')[-2] + '_sbvf/logs/'

directories = [output_task, output_loss, output_stats, output_models, output_val, output_logs]

for dir in directories:
    try:
        os.makedirs(dir)  
        
    except FileExistsError:
        pass

history.to_csv(output_loss + task + '.csv', sep=',', encoding='utf-8', header='true')

plot_history(history, output_loss, True, task)

torch.save(model_1.state_dict(), output_models + task + '.pt')
np.save(output_logs + 'sbvfs.npy', VFs)
np.save(output_logs + 'w_virt.npy', w_virt)
#joblib.dump([VFs, W_virt], 'sbvfs.pkl')
if train_generator.std == True:
    joblib.dump(train_generator.scaler_x, output_models + task + '-scaler_x.pkl')

# Deleting temp folder
try:
    shutil.rmtree('temp/')    
except FileNotFoundError:
    pass

# ---------------------------------
#    Library and function imports
# ---------------------------------
import re
from constants import *
from functions import custom_loss, data_sampling, load_dataframes, select_features_multi, standardize_data, plot_history, standardize_
from functions import EarlyStopping
from sklearn.model_selection import GroupShuffleSplit

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
from scipy.optimize import fsolve

# -------------------------------
#        Class definitions
# -------------------------------
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, deformation, stress, force, coord, list_IDs, batch_size, shuffle):
        super().__init__()
        self.X = deformation.iloc[list_IDs]
        self.y = stress.iloc[list_IDs]
        self.f = force.iloc[list_IDs]
        self.coord = coord[['x','y']].iloc[list_IDs]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.list_IDs = list_IDs
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = np.array([self.indexes[index]+i for i in range(self.batch_size)])

        # Generate data
        X, y, f, coord = self.__data_generation(np.random.permutation(indexes))

        return X, y, f, coord

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(0, len(self.list_IDs), self.batch_size)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    # def standardize(self):
    #     idx = self.X.index
    #     self.X, self.y, self.scaler_x, self.scaler_y = standardize_data(self.X, self.y)
    #     self.f, self.scaler_f = standardize_(self.f)
    #     #self.coord, self.scaler_coord = standardize_(self.coord[['x','y']])

    #     self.X = pd.DataFrame(self.X, index=idx)
    #     self.y = pd.DataFrame(self.y, index=idx)
    #     self.f = pd.DataFrame(self.f, index=idx)
    #     #self.coord = pd.DataFrame(self.coord, index=idx)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        X = np.asarray(self.X.iloc[list_IDs_temp], dtype=np.float32)
        y = np.asarray(self.y.iloc[list_IDs_temp], dtype=np.float32)
        f = np.asarray(self.f.iloc[list_IDs_temp], dtype=np.float32)
        coord = np.asarray(self.coord.iloc[list_IDs_temp], dtype=np.float32)
        return X, y, f, coord

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_hidden_layers=1):
        super(NeuralNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size  = hidden_size
        self.n_hidden_layers = n_hidden_layers
        self.output_size = output_size

        self.layers = nn.ModuleList()

        for i in range(self.n_hidden_layers):
            if i == 0:
                in_ = self.input_size
            else:
                in_ = self.hidden_size

            self.layers.append(torch.nn.Linear(in_, self.hidden_size, bias=True))

        self.layers.append(torch.nn.Linear(self.hidden_size, self.output_size, bias=True))

        self.activation = torch.nn.PReLU(self.hidden_size)

    def forward(self, x):

        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
            
        return self.layers[-1](x)

# -------------------------------
#       Method definitions
# -------------------------------

def func(x, H):

    A = [x[0]**2 - H[0]]
    A.append(x[0] * x[1] - H[1])
    A.append(x[0] * x[3] - H[2])
    A.append(x[1]**2 + x[2]**2 - H[3])
    A.append(x[1] * x[3] + x[2] * x[4] - H[4])
    A.append(x[5]**2 - H[5])
    
    return A
    
# As in: https://gist.github.com/MasanoriYamada/d1d8ca884d200e73cca66a4387c7470a
def get_batch_jacobian(net, x, to):
    # noutputs: total output dim (e.g. net(x).shape(b,1,4,4) noutputs=1*4*4
    # b: batch
    # i: in_dim
    # o: out_dim
    # ti: total input dim
    # to: total output dim
    x_batch = x.shape[0]
    x_shape = x.shape[1:]
    x = x.unsqueeze(1)  # b, 1 ,i
    x = x.repeat(1, to, *(1,)*len(x.shape[2:]))  # b * to,i  copy to o dim
    x.requires_grad_(True)
    tmp_shape = x.shape
    y = net(x.reshape(-1, *tmp_shape[2:]))  # x.shape = b*to,i y.shape = b*to,to
    y_shape = y.shape[1:]  # y.shape = b*to,to
    y = y.reshape(x_batch, to, to)  # y.shape = b,to,to
    input_val = torch.eye(to).reshape(1, to, to).repeat(x_batch, 1, 1)  # input_val.shape = b,to,to  value is (eye)
    y.backward(input_val)  # y.shape = b,to,to
    return x.grad.reshape(x_batch, *y_shape, *x_shape).data  # x.shape = b,o,i

def get_v_fields(cent_x, cent_y, x, y):

    # Auxialiary vectors
    zeros_ = torch.zeros_like(x)

    ones = torch.ones(size=(batch_size,), dtype=torch.float32)
    zeros = torch.zeros(size=(batch_size,), dtype=torch.float32)

    # Virtual displacement fields
    virtual_disp = {
        1: torch.stack([x/LENGTH, zeros_], 1),
        2: torch.stack([zeros_, y/LENGTH], 1),
        3: torch.stack([zeros_, y*(torch.square(x)-x*LENGTH)/LENGTH**3], 1),
        4: torch.stack([zeros_, torch.sin(x*math.pi/LENGTH)*torch.sin(y*math.pi/LENGTH)], 1),
        5: torch.stack([torch.sin(y*math.pi/LENGTH) * torch.sin(x*math.pi/LENGTH), zeros_], 1),
        6: torch.stack([x*y*(x-LENGTH)/LENGTH**3,zeros_], 1),
        7: torch.stack([torch.square(x)*(LENGTH-x)*torch.sin(math.pi*y/LENGTH)/LENGTH**3,zeros_], 1),
        8: torch.stack([zeros_, (LENGTH**3-x**3)*torch.sin(math.pi*y/LENGTH)/LENGTH**3], 1),
        9: torch.stack([(x*LENGTH**2-x**3)*torch.sin(y*math.pi/LENGTH)/LENGTH**3, zeros_], 1),
        10: torch.stack([(x*y*(x-LENGTH)/LENGTH**2)*torch.sin(y*math.pi/LENGTH), zeros_], 1),
        11: torch.stack([zeros_, (x*y*(y-LENGTH)/LENGTH**2)*torch.sin(x*math.pi/LENGTH)], 1),
        12: torch.stack([zeros_, x*y*(y-LENGTH)/LENGTH**3], 1),
        13: torch.stack([y**2*torch.sin(x*math.pi/LENGTH)/LENGTH**2, zeros_], 1),
        14: torch.stack([zeros_, x**2*torch.sin(y*math.pi/LENGTH)/LENGTH**2], 1),
        15: torch.stack([(x*y*(x-LENGTH)/LENGTH**2)*np.sin(x**2*y**2/LENGTH**4), zeros_], 1),
        16: torch.stack([torch.sin(x*math.pi/LENGTH)/LENGTH, zeros_], 1),
        17: torch.stack([zeros_, torch.sin(y*math.pi/LENGTH)/LENGTH], 1),
        18: torch.stack([torch.sin(x**3*math.pi/LENGTH**3)/LENGTH**3, zeros_], 1),
        19: torch.stack([zeros_, torch.sin(y**3*math.pi/LENGTH**3)/LENGTH**3], 1)
    }    

    # Defining virtual strain fields
    virtual_strain = {
        1:torch.stack([ones/LENGTH, zeros, zeros], 1),
        2:torch.stack([zeros, ones/LENGTH, zeros], 1),
        3:torch.stack([zeros, (torch.square(cent_x)-cent_x*LENGTH)/LENGTH**3, cent_y*(2*cent_x-LENGTH)/LENGTH**3], 1),
        4:torch.stack([zeros, (math.pi/LENGTH)*torch.sin(cent_x*math.pi/LENGTH)*torch.cos(cent_y*math.pi/LENGTH), (math.pi/LENGTH)*torch.cos(cent_x*math.pi/LENGTH)*torch.sin(cent_y*math.pi/LENGTH)],1),
        5:torch.stack([(math.pi/LENGTH)*torch.cos(cent_x*math.pi/LENGTH)*torch.sin(cent_y*math.pi/LENGTH), zeros, (math.pi/LENGTH)*torch.sin(cent_x*math.pi/LENGTH)*torch.cos(cent_y*math.pi/LENGTH)],1),
        6:torch.stack([(2*cent_x*cent_y-cent_y*LENGTH)/LENGTH**3, zeros, (torch.square(cent_x)-cent_x*LENGTH)/LENGTH**3], 1),
        7:torch.stack([(2*cent_x*LENGTH-3*torch.square(cent_x))*torch.sin(math.pi*cent_y/LENGTH)/LENGTH**3, zeros, (torch.square(cent_x)*LENGTH-cent_x**3)*torch.cos(math.pi*cent_y/LENGTH)*math.pi/LENGTH**4], axis=1),
        8:torch.stack([zeros, (LENGTH**3-cent_x**3)*math.pi*torch.cos(math.pi*cent_y/LENGTH)/LENGTH**4, (-3*torch.square(cent_x)/LENGTH**3)*torch.sin(math.pi*cent_y/LENGTH)], axis=1),
        9:torch.stack([(LENGTH**2-3*torch.square(cent_x))*torch.sin(math.pi*cent_y/LENGTH)/LENGTH**3, zeros, (cent_x*LENGTH**2-cent_x**3)*torch.cos(math.pi*cent_y/LENGTH)*math.pi/LENGTH**4],1),
        10:torch.stack([(2*cent_x*cent_y-cent_y*LENGTH)*torch.sin(cent_y*math.pi/LENGTH)/LENGTH**2, zeros, (torch.square(cent_x)-cent_x*LENGTH)*torch.sin(cent_y*math.pi/LENGTH)/LENGTH**2+(cent_x*cent_y*(cent_x-LENGTH))*torch.cos(cent_y*math.pi/LENGTH)*math.pi/LENGTH**3], 1),
        11:torch.stack([zeros, (2*cent_x*cent_y-cent_x*LENGTH)*torch.sin(cent_x*math.pi/LENGTH)/LENGTH**2, (torch.square(cent_y)-cent_y*LENGTH)*torch.sin(cent_x*math.pi/LENGTH)/LENGTH**2+(cent_x*cent_y*(cent_y-LENGTH))*torch.cos(cent_x*math.pi/LENGTH)*math.pi/LENGTH**3], 1),
        12:torch.stack([zeros, cent_x*(2*cent_y-LENGTH)/LENGTH**3, cent_y*(cent_y-LENGTH)/LENGTH**3], 1),
        13: torch.stack([cent_y**2*torch.cos(cent_x*math.pi/LENGTH)*math.pi/LENGTH**3, zeros, 2*cent_y*torch.sin(cent_x*math.pi/LENGTH)/LENGTH**2], 1),
        14: torch.stack([zeros, cent_x**2*torch.cos(cent_y*math.pi/LENGTH)*math.pi/LENGTH**3, 2*cent_x*torch.sin(cent_y*math.pi/LENGTH)/LENGTH**2], 1),
        15: torch.stack([((2*cent_x*cent_y-cent_y*LENGTH)*torch.sin(cent_x**2*cent_y**2/LENGTH**4)/LENGTH**2) + (cent_x*cent_y*(cent_x-LENGTH)*torch.cos(cent_x**2*cent_y**2/LENGTH**4)*2*cent_x*cent_y**2/LENGTH**6), zeros, ((cent_x**2-cent_x*LENGTH)*torch.sin(cent_x**2*cent_y**2/LENGTH**4)/LENGTH**2) + (cent_x*cent_y*(cent_x-LENGTH)*torch.cos(cent_x**2*cent_y**2/LENGTH**4)*2*cent_x**2*cent_y/LENGTH**6)],1),
        16: torch.stack([torch.cos(cent_x*math.pi/LENGTH)*math.pi/LENGTH**2, zeros, zeros],1),
        17: torch.stack([zeros, torch.cos(cent_y*math.pi/LENGTH)*math.pi/LENGTH**2, zeros],1),
        18: torch.stack([torch.cos(cent_x**3*math.pi/LENGTH**3)*3*cent_x**2*math.pi/LENGTH**6, zeros, zeros], 1),
        19: torch.stack([zeros, torch.cos(cent_y**3*math.pi/LENGTH**3)*3*cent_y**2*math.pi/LENGTH**6, zeros], 1)
    }

    # Total number of virtual fields
    total_vfs = len(virtual_disp.keys())

    # Converting virtual displacement/strain fields dictionaries into a tensors
    v_disp = torch.stack(list(virtual_disp.values()))
    v_strain = torch.stack(list(virtual_strain.values()))

    return total_vfs, v_disp, v_strain

def init_weights(m):
    '''
    Performs the weight initialization of a neural network

    Parameters
    ----------
    m : NeuralNetwork object
        Neural network model, instance of NeuralNework class
    '''
    if isinstance(m, nn.Linear) and (m.bias != None):
        torch.nn.init.kaiming_uniform_(m.weight)
        #m.bias.data.fill_(0.001)
        torch.nn.init.ones_(m.bias)

def train_loop(dataloader, model, loss_fn, optimizer):
    '''
    Custom loop for neural network training, using mini-batches

    '''
    num_batches = len(dataloader)
    losses = torch.zeros(num_batches)
    v_work_real = torch.zeros(num_batches)

    for batch in range(num_batches):
        # Extracting variables for training
        X_train, y_train, f_train, coord = dataloader[batch]
        
        # Converting to pytorch tensors
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        f_train = torch.tensor(f_train, dtype=torch.float32) + 0.0
        coord = torch.tensor(coord, dtype=torch.float32)
         
        # Extracting centroid coordinates
        cent_x = coord[:,0]
        cent_y = coord[:,1]

        # Defining surface coordinates where load is applied
        n_surf_nodes = int((batch_size**0.5) + 1)
        x = LENGTH * torch.ones(n_surf_nodes, dtype=torch.float32)
        y = torch.tensor(np.linspace(0, LENGTH, n_surf_nodes), dtype=torch.float32)

        total_vfs, v_disp, v_strain = get_v_fields(cent_x, cent_y, x, y)

        # Computing prediction
        pred = model(X_train)

        # Calculating global force from stress predictions
        #global_f_pred = torch.sum(pred * ELEM_AREA * ELEM_THICK / LENGTH, 0)[:-1]

        # dsde = get_batch_jacobian(model, X_train, N_OUTPUTS)[:,:,-3:]
        # ltri = torch.stack([torch.tril(dsde[i]).flatten() for i in range(batch_size)])
        # ltri = torch.stack([ltri[i][ltri[i].nonzero()] for i in range(batch_size)]).detach().numpy()
        
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
        

        # Computing the internal virtual works
        int_work = -ELEM_THICK * torch.sum(torch.reshape(torch.sum(pred * v_strain[:] * ELEM_AREA, -1, keepdims=True),[total_vfs,batch_size//batch_size,batch_size,1]),2)
        int_work = torch.reshape(int_work,[-1,1])

        # Computing the internal virtual work coming from stress labels (results checking only)
        int_work_real = -ELEM_THICK * torch.sum(torch.reshape(torch.sum(y_train * v_strain[:] * ELEM_AREA, -1, keepdims=True),[total_vfs,batch_size//batch_size,batch_size,1]),2)        
        int_work_real = torch.reshape(int_work_real,[-1,1])

        # Extracting global force components and filtering duplicate values
        f = f_train[:,:2][::batch_size,:]
        
        # Computing external virtual work
        ext_work = torch.mean(torch.sum(f*v_disp[:],-1,keepdims=True),1) 

        # Computing losses        
        loss = loss_fn(int_work, ext_work)
        cost = loss_fn(int_work_real, ext_work)

        # Backpropagation and weight's update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Saving loss values
        losses[batch] = loss
        v_work_real[batch] = cost

        print('\r>Train: %d/%d' % (batch + 1, num_batches), end='')

    return losses, v_work_real, total_vfs

def test_loop(dataloader, model, loss_fn):
    
    num_batches = len(dataloader)
    test_losses = torch.zeros(num_batches)

    model.eval()
    with torch.no_grad():

        for batch in range(num_batches):

            # Extracting variables for testing
            X_test, y_test, f_test, coord = dataloader[batch]
            
            # Converting to pytorch tensors
            X_test = torch.tensor(X_test, dtype=torch.float32)
            y_test = torch.tensor(y_test, dtype=torch.float32)
            f_test = torch.tensor(f_test, dtype=torch.float32) + 0.0
            coord = torch.tensor(coord, dtype=torch.float32)
            
            # Extracting centroid coordinates
            cent_x = coord[:,0]
            cent_y = coord[:,1]

            # Defining surface coordinates where load is applied
            n_surf_nodes = int((batch_size**0.5) + 1)
            x = LENGTH * torch.ones(n_surf_nodes, dtype=torch.float32)
            y = torch.tensor(np.linspace(0, LENGTH, n_surf_nodes), dtype=torch.float32)

            total_vfs, v_disp, v_strain = get_v_fields(cent_x, cent_y, x, y)

            pred = model(X_test)

            # Computing the internal virtual works
            int_work = -ELEM_THICK * torch.sum(torch.reshape(torch.sum(pred * v_strain[:] * ELEM_AREA, -1, keepdims=True),[total_vfs,batch_size//batch_size,batch_size,1]),2)
            int_work = torch.reshape(int_work,[-1,1])

            # Computing the internal virtual work coming from stress labels (results checking only)
            int_work_real = -ELEM_THICK * torch.sum(torch.reshape(torch.sum(y_test * v_strain[:] * ELEM_AREA, -1, keepdims=True),[total_vfs,batch_size//batch_size,batch_size,1]),2)        
            int_work_real = torch.reshape(int_work_real,[-1,1])

            # Extracting global force components and filtering duplicate values
            f = f_test[:,:2][::batch_size,:]
            
            # Computing external virtual work
            ext_work = torch.mean(torch.sum(f*v_disp[:],-1,keepdims=True),1) 

            # Computing losses        
            test_loss = loss_fn(int_work, ext_work)

            test_losses[batch] = test_loss

            print('\r>Test: %d/%d' % (batch + 1, num_batches), end='')

    return test_losses

# -------------------------------
#           Main script
# -------------------------------

# Specifying random seed
random.seed(SEED)

# Loading data
df_list, _ = load_dataframes(TRAIN_MULTI_DIR)

# Sampling data pass random seed for random sampling
sampled_dfs = data_sampling(df_list, DATA_SAMPLES)

# Merging training data
data = pd.concat(sampled_dfs, axis=0, ignore_index=True)

# Reorganizing dataset by time increment, subsequent grouping by tag and final shuffling
data_by_t = [df for _, df in data.groupby(['t'])]
random.shuffle(data_by_t)
data_by_tag = [[df for _, df in group.groupby(['tag'])] for group in data_by_t]
random.shuffle(data_by_tag)
data_by_batches = list(itertools.chain(*data_by_tag))
random.shuffle(data_by_batches)

batch_size = len(data_by_batches[0])

# Concatenating data groups
data = pd.concat(data_by_batches).reset_index(drop=True)

# Selecting model features
X, y, f, coord = select_features_multi(data)

# Performing test/train split
partition = {"train": None, "test": None}
partition['train'], partition['test'] = next(GroupShuffleSplit(test_size=TEST_SIZE, n_splits=2, random_state = SEED).split(data, groups=data['t']))

# Preparing data generators for mini-batch training
train_generator = DataGenerator(X, y, f, coord, partition["train"], batch_size, True)
test_generator = DataGenerator(X, y, f, coord, partition['test'], batch_size, True)

# Model variables
N_INPUTS = X.shape[1]
N_OUTPUTS = y.shape[1]

N_UNITS = 8
H_LAYERS = 1

model = NeuralNetwork(N_INPUTS, N_OUTPUTS, N_UNITS, H_LAYERS)
model.apply(init_weights)

# Training variables
epochs = 75

# Optimization variables
learning_rate = 0.1
loss_fn = custom_loss
f_loss = torch.nn.MSELoss()

optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.2, threshold=1e-3)

# Container variables for history purposes
train_loss = []
v_work = []
val_loss = []
epochs_ = []

# Initializing the early_stopping object
#early_stopping = EarlyStopping(patience=6, verbose=True)

for t in range(epochs):

    print('\r--------------------\nEpoch [%d/%d]' % (t + 1, epochs))

    epochs_.append(t+1)
    
    start_epoch = time.time()

    #Shuffling batches
    train_generator.on_epoch_end()
    
    # Train loop
    start_train = time.time()
    batch_losses, batch_v_work, n_vfs = train_loop(train_generator, model, loss_fn, optimizer)
    
    train_loss.append(torch.mean(batch_losses).item())
    v_work.append(torch.mean(batch_v_work).item())
    
    end_train = time.time()
    
    #Apply learning rate scheduling if defined
    try:
        scheduler.step(train_loss[t])       
        print('. t_loss: %.3f -> lr: %.3e // [v_work] -> %.3e -- %.3fs' % (train_loss[t], scheduler._last_lr[0], v_work[t], end_train - start_train))
    except:
        print('. loss: %.3f // [v_work] -> %.3e -- %.3fs' % (train_loss[t], v_work[t], end_train - start_train))

    # Test loop
    start_test = time.time()

    batch_val_losses = test_loop(test_generator, model, loss_fn)

    val_loss.append(torch.mean(batch_val_losses).item())

    end_test = time.time()

    print('. v_loss: %.3f -- %.3fs' % (val_loss[t], end_test - start_test))

    end_epoch = time.time()

    # Check validation loss for early stopping
    # early_stopping(val_loss[t], model)

    # if early_stopping.early_stop:
    #     print("Early stopping")
    #     break

print("Done!")

# load the last checkpoint with the best model
# model.load_state_dict(torch.load('checkpoint.pt'))

epochs_ = np.reshape(np.array(epochs_), (len(epochs_),1))
train_loss = np.reshape(np.array(train_loss), (len(train_loss),1))
val_loss = np.reshape(np.array(val_loss), (len(val_loss),1))

history = pd.DataFrame(np.concatenate([epochs_, train_loss, val_loss], axis=1), columns=['epoch','loss','val_loss'])

task = r'[%i-%ix%i-%i]-%s-%i-VFs' % (N_INPUTS, N_UNITS, H_LAYERS, N_OUTPUTS, TRAIN_MULTI_DIR.split('/')[-2], n_vfs)

import os
output_task = 'outputs/' + TRAIN_MULTI_DIR.split('/')[-2]
output_loss = 'outputs/' + TRAIN_MULTI_DIR.split('/')[-2] + '/loss/'
output_prints = 'outputs/' + TRAIN_MULTI_DIR.split('/')[-2] + '/prints/'
output_models = 'outputs/' + TRAIN_MULTI_DIR.split('/')[-2] + '/models/'
output_val = 'outputs/' + TRAIN_MULTI_DIR.split('/')[-2] + '/val/'

directories = [output_task, output_loss, output_prints, output_models, output_val]

for dir in directories:
    try:
        os.makedirs(dir)  
        
    except FileExistsError:
        pass

history.to_csv(output_loss + task + '.csv', sep=',', encoding='utf-8', header='true')

plot_history(history, output_prints, True, task)

torch.save(model.state_dict(), output_models + task + '.pt')
torch.save(model.state_dict(), 'models/ann_torch/model_1.pt')
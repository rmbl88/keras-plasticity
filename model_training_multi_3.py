# ---------------------------------
#    Library and function imports
# ---------------------------------
import re
from constants import *
from functions import custom_loss, data_sampling, load_dataframes, select_features_multi, standardize_data, plot_history, standardize_
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

    def standardize(self):
        idx = self.X.index
        self.X, self.y, self.scaler_x, self.scaler_y = standardize_data(self.X, self.y)
        self.f, self.scaler_f = standardize_(self.f)
        #self.coord, self.scaler_coord = standardize_(self.coord[['x','y']])

        self.X = pd.DataFrame(self.X, index=idx)
        self.y = pd.DataFrame(self.y, index=idx)
        self.f = pd.DataFrame(self.f, index=idx)
        #self.coord = pd.DataFrame(self.coord, index=idx)

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

def get_v_fields(cent_x, cent_y, x, y):

    # Defining surface coordinates where load is applied
    # n_surf_nodes = int((batch_size**0.5) + 1)
    # x = LENGTH * torch.ones(n_surf_nodes, dtype=torch.float32)
    # y = torch.tensor(np.linspace(0, LENGTH, n_surf_nodes), dtype=torch.float32)

    # Auxialiary vectors
    #zeros_ = torch.zeros(n_surf_nodes, dtype=torch.float32)
    zeros_ = torch.zeros_like(x)

    ones = torch.ones(size=(batch_size,), dtype=torch.float32)
    zeros = torch.zeros(size=(batch_size,), dtype=torch.float32)

    # Virtual displacement fields
    virtual_disp = {
        1: torch.stack([x/LENGTH, zeros_], 1),
        2: torch.stack([zeros_, y/LENGTH], 1),
        3: torch.stack([(x*y/LENGTH**3), (x*y/LENGTH**3)], 1),
        4: torch.stack([zeros_, y*(torch.square(x)-x*LENGTH)/LENGTH**3], 1),
        5: torch.stack([zeros_, torch.sin(x*math.pi/LENGTH)*torch.sin(y*math.pi/LENGTH)], 1),
        6: torch.stack([torch.sin(y*math.pi/LENGTH) * torch.sin(x*math.pi/LENGTH), zeros_], 1),
        7: torch.stack([x*y*(x-LENGTH)/LENGTH**3,zeros_], 1),
        8: torch.stack([torch.square(x)*(LENGTH-x)*torch.sin(math.pi*y/LENGTH)/LENGTH**3,zeros_], 1),
        9: torch.stack([zeros_, (LENGTH**3-x**3)*torch.sin(math.pi*y/LENGTH)/LENGTH**3], 1),
        10: torch.stack([(x*y*(x-LENGTH)/LENGTH**2)*torch.sin(y*math.pi/LENGTH), zeros_], 1),
        11: torch.stack([zeros_, (x*y*(y-LENGTH)/LENGTH**2)*torch.sin(x*math.pi/LENGTH)], 1),
        12: torch.stack([zeros_, x*y*(y-LENGTH)/LENGTH**3], 1),
        13: torch.stack([(x*y*(x-LENGTH)/LENGTH**2)*torch.sin(y*math.pi/LENGTH), (x*y*(y-LENGTH)/LENGTH**2)*torch.sin(x*math.pi/LENGTH)],1),
        14: torch.stack([y**2*torch.sin(x*math.pi/LENGTH)/LENGTH**2, zeros_], 1),
        15: torch.stack([y**2*torch.sin(x*math.pi/LENGTH)/LENGTH**2, x**2*torch.sin(y*math.pi/LENGTH)/LENGTH**2], 1),
        16: torch.stack([(x*y*(x-LENGTH)/LENGTH**2)*np.sin(x**2*y**2/LENGTH**3), zeros_], 1)
    }    

    # Defining virtual strain fields
    virtual_strain = {
        1:torch.stack([ones/LENGTH, zeros, zeros], 1),
        2:torch.stack([zeros, ones/LENGTH, zeros], 1),
        3:torch.stack([cent_y/LENGTH**3, cent_x/LENGTH**3, (cent_x+cent_y)/LENGTH**3], 1),
        4:torch.stack([zeros, (torch.square(cent_x)-cent_x*LENGTH)/LENGTH**3, cent_y*(2*cent_x-LENGTH)/LENGTH**3], 1),
        5:torch.stack([zeros, (math.pi/LENGTH)*torch.sin(cent_x*math.pi/LENGTH)*torch.cos(cent_y*math.pi/LENGTH), (math.pi/LENGTH)*torch.cos(cent_x*math.pi/LENGTH)*torch.sin(cent_y*math.pi/LENGTH)],1),
        6:torch.stack([(math.pi/LENGTH)*torch.cos(cent_x*math.pi/LENGTH)*torch.sin(cent_y*math.pi/LENGTH), zeros, (math.pi/LENGTH)*torch.sin(cent_x*math.pi/LENGTH)*torch.cos(cent_y*math.pi/LENGTH)],1),
        7:torch.stack([(2*cent_x*cent_y-cent_y*LENGTH)/LENGTH**3, zeros, (torch.square(cent_x)-cent_x*LENGTH)/LENGTH**3], 1),
        8:torch.stack([(2*cent_x*LENGTH-3*torch.square(cent_x))*torch.sin(math.pi*cent_y/LENGTH)/LENGTH**3, zeros, (torch.square(cent_x)*LENGTH-cent_x**3)*torch.sin(math.pi*cent_y/LENGTH)/LENGTH**4], axis=1),
        9:torch.stack([zeros, (LENGTH**3-cent_x**3)*math.pi*torch.cos(math.pi*cent_y/LENGTH)/LENGTH**4, (-3*torch.square(cent_x)/LENGTH**3)*torch.sin(math.pi*cent_y/LENGTH)], axis=1),
        10:torch.stack([(2*cent_x*cent_y-cent_y*LENGTH)*torch.sin(cent_y*math.pi/LENGTH)/LENGTH**2, zeros, (torch.square(cent_x)-cent_x*LENGTH)*torch.sin(cent_y*math.pi/LENGTH)/LENGTH**2+(cent_x*cent_y*(cent_x-LENGTH))*torch.cos(cent_y*math.pi/LENGTH)*math.pi/LENGTH**3], 1),
        11:torch.stack([zeros, (2*cent_x*cent_y-cent_x*LENGTH)*torch.sin(cent_x*math.pi/LENGTH)/LENGTH**2, (torch.square(cent_y)-cent_y*LENGTH)*torch.sin(cent_x*math.pi/LENGTH)/LENGTH**2+(cent_x*cent_y*(cent_y-LENGTH))*torch.cos(cent_x*math.pi/LENGTH)*math.pi/LENGTH**3], 1),
        12:torch.stack([zeros, cent_x*(2*cent_y-LENGTH)/LENGTH**3, cent_y*(cent_y-LENGTH)/LENGTH**3], 1),
        13:torch.stack([(2*cent_x*cent_y-cent_y*LENGTH)*torch.sin(cent_y*math.pi/LENGTH)/LENGTH**2, (2*cent_x*cent_y-cent_x*LENGTH)*torch.sin(cent_x*math.pi/LENGTH)/LENGTH**2, ((torch.square(cent_x)-cent_x*LENGTH)*torch.sin(cent_y*math.pi/LENGTH)/LENGTH**2+(cent_x*cent_y*(cent_x-LENGTH))*torch.cos(cent_y*math.pi/LENGTH)*math.pi/LENGTH**3) + ((torch.square(cent_y)-cent_y*LENGTH)*torch.sin(cent_x*math.pi/LENGTH)/LENGTH**2+(cent_x*cent_y*(cent_y-LENGTH))*torch.cos(cent_x*math.pi/LENGTH)*math.pi/LENGTH**3)], 1),
        14: torch.stack([cent_y**2*torch.cos(cent_x*math.pi/LENGTH)*math.pi/LENGTH**3, zeros, 2*cent_y*torch.sin(cent_x*math.pi/LENGTH)/LENGTH**2], 1),
        15: torch.stack([cent_y**2*torch.cos(cent_x*math.pi/LENGTH)*math.pi/LENGTH**3, cent_x**2*torch.cos(cent_y*math.pi/LENGTH)*math.pi/LENGTH**3, (2*cent_y*torch.sin(cent_x*math.pi/LENGTH)/LENGTH**2) + (2*cent_x*torch.sin(cent_y*math.pi/LENGTH)/LENGTH**2)], 1),
        16: torch.stack([((2*cent_x*cent_y-cent_y*LENGTH)*torch.sin(cent_x**2*cent_y**2/LENGTH**3)/LENGTH**2) + (cent_x*cent_y*(cent_x-LENGTH)*torch.cos(cent_x**2*cent_y**2/LENGTH**3)*2*cent_x*cent_y**2/LENGTH**5), zeros, ((cent_x**2-cent_x*LENGTH)*torch.sin(cent_x**2*cent_y**2/LENGTH**3)/LENGTH**2) + (cent_x*cent_y*(cent_x-LENGTH)*torch.cos(cent_x**2*cent_y**2/LENGTH**3)*2*cent_x**2*cent_y/LENGTH**5)],1)
        
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

        # Computing the internal virtual works
        int_work = ELEM_THICK * torch.sum(torch.reshape(torch.sum(pred * v_strain[:] * ELEM_AREA, -1, keepdims=True),[total_vfs,batch_size//batch_size,batch_size,1]),2)
        int_work = torch.reshape(int_work,[-1,1])

        # Computing the internal virtual work coming from stress labels (results checking only)
        int_work_real = ELEM_THICK * torch.sum(torch.reshape(torch.sum(y_train * v_strain[:] * ELEM_AREA, -1, keepdims=True),[total_vfs,batch_size//batch_size,batch_size,1]),2)        
        int_work_real = torch.reshape(int_work_real,[-1,1])

        # Extracting global force components and filtering duplicate values
        f = f_train[:,:2][::batch_size,:]
        
        # Computing external virtual work
        ext_work = torch.mean(torch.sum(f*v_disp[:],-1,keepdims=True),1) 

        # Computing losses        
        loss = loss_fn(int_work, ext_work)
        cost = loss_fn(int_work_real, ext_work)

        # Backpropagation and parametersgit st update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Saving loss values
        losses[batch] = loss
        v_work_real[batch] = cost

        # print('\rEpoch [%d/%d] Batch: %d/%d' % (epoch + 1, epochs, batch+1,size), end='')

        print('\r>Train: %d/%d' % (batch + 1, num_batches), end='')

    return losses, v_work_real

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
            int_work = ELEM_THICK * torch.sum(torch.reshape(torch.sum(pred * v_strain[:] * ELEM_AREA, -1, keepdims=True),[total_vfs,batch_size//batch_size,batch_size,1]),2)
            int_work = torch.reshape(int_work,[-1,1])

            # Computing the internal virtual work coming from stress labels (results checking only)
            int_work_real = ELEM_THICK * torch.sum(torch.reshape(torch.sum(y_test * v_strain[:] * ELEM_AREA, -1, keepdims=True),[total_vfs,batch_size//batch_size,batch_size,1]),2)        
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

# Specifying device for training
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'

# Loading data
df_list, _ = load_dataframes(TRAIN_MULTI_DIR)

# Sampling data pass random seed for random sampling
sampled_dfs = data_sampling(df_list, DATA_SAMPLES)

# Merging training data
data = pd.concat(sampled_dfs, axis=0, ignore_index=True)

# Reorganizing dataset by time increment and applying first shuffling
data_groups = [df for _, df in data.groupby(['t'])]
random.shuffle(data_groups)

data = pd.concat(data_groups).reset_index(drop=True)

X, y, f, coord = select_features_multi(data)

partition = {"train": None, "test": None}

partition['train'], partition['test'] = next(GroupShuffleSplit(test_size=TEST_SIZE, n_splits=2, random_state = SEED).split(data, groups=data['t']))



N_INPUTS = X.shape[1]
N_OUTPUTS = y.shape[1]

N_UNITS = 16

model = NeuralNetwork(N_INPUTS, N_OUTPUTS, N_UNITS, 2)
model.apply(init_weights)

# Model variables
learning_rate = 0.1

#batch_size = len(data_groups[0])
batch_size = 36

epochs = 100

loss_fn = torch.nn.MSELoss(reduction='mean')
#loss_fn = torch.nn.HuberLoss(reduction='mean')

#optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)
#optimizer = torch.optim.Adagrad(params=model.parameters(), lr=learning_rate)
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.2, threshold=1e-3)
#scheduler =  torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=0.0001, last_epoch=-1)

# Preparing training generator for mini-batch training
#train_generator = DataGenerator(X, y, f, coord, X.index.tolist(), batch_size, True)
train_generator = DataGenerator(X, y, f, coord, partition["train"], batch_size, True)
test_generator = DataGenerator(X, y, f, coord, partition['test'], batch_size, False)

train_loss = np.zeros(shape=(epochs,1), dtype=np.float32)
v_work = np.zeros(shape=(epochs,1), dtype=np.float32)
val_loss = np.zeros(shape=(epochs,1), dtype=np.float32)

epochs_ = np.arange(0,epochs).reshape(epochs,1)

for t in range(epochs):

    print('\r--------------------\nEpoch [%d/%d]' % (t + 1, epochs))
    
    start_epoch = time.time()

    #Shuffling batches
    train_generator.on_epoch_end()
    
    # Train loop
    start_train = time.time()
    batch_losses, batch_v_work = train_loop(train_generator, model, loss_fn, optimizer)
    
    train_loss[t]=torch.mean(batch_losses).detach().numpy()
    v_work = torch.mean(batch_v_work).detach().numpy()
    
    #Apply learning rate scheduling
    scheduler.step(train_loss[t])
    #scheduler.step()

    end_train = time.time()
        
    print('.t_loss: %.3f -> lr: %.3e // [v_work] -> %.3e -- %.3f s' % (train_loss[t][0], scheduler._last_lr[0], v_work, end_train - start_train))
    #print('. loss: %.3f // [cost] -> %.3e -- %.3f s' % (train_loss[t][0], cost, time.time()-start_time))

    # Test loop
    start_test = time.time()
    batch_val_losses = test_loop(test_generator, model, loss_fn)

    val_loss[t]=torch.mean(batch_val_losses).detach().numpy()

    end_test = time.time()

    print('.v_loss: %.3f -- %.3f s' % (val_loss[t][0], end_test - start_test))

    end_epoch = time.time()

print("Done!")

history = pd.DataFrame(np.concatenate([epochs_, train_loss, val_loss], axis=1), columns=['epoch','loss','val_loss'])

plot_history(history, True)

torch.save(model.state_dict(), 'models/ann_torch/model_1')


# for p in model.parameters():
#     if p.grad is not None:
#         print(p.grad.data)
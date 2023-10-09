# ---------------------------------
#    Library and function imports
# ---------------------------------
import re

import joblib
from sklearn.utils import shuffle
from constants import *
from functions import custom_loss, data_sampling, load_dataframes, select_features_multi, standardize_data, plot_history
from functions import EarlyStopping
from sklearn.model_selection import GroupShuffleSplit
import copy
from re import S
from torch import nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pandas as pd
from sklearn import preprocessing
import random
import numpy as np
import math
import torch
import time
import itertools
from scipy.optimize import fsolve
import wandb
from torch.autograd.functional import jacobian
import geotorch


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

        self.activation_h = torch.nn.PReLU(self.hidden_size)
        self.activation_o = torch.nn.PReLU(self.output_size)

    def forward(self, x):

        for layer in self.layers[:-1]:
            
            x = self.activation_h(layer(x))
            
        #return self.layers[-1](x)
        return self.activation_o(self.layers[-1](x))
    
    def compute_l1_loss(self, w):
      return (torch.abs(w)**0.25).sum()

# -------------------------------
#       Method definitions
# -------------------------------

def get_v_fields(cent_x, cent_y, x, y, dir):

    # Auxialiary vectors
    zeros_ = torch.zeros_like(x)

    ones = torch.ones(size=(batch_size,), dtype=torch.float64)
    zeros = torch.zeros(size=(batch_size,), dtype=torch.float64)

    if dir == 1:
        # Virtual displacement fields
        virtual_disp = {
            1: torch.stack([x/LENGTH, zeros_], 1),
            2: torch.stack([zeros_,torch.sin(2*y*math.pi/(3*LENGTH))], 1),
            31: torch.stack([zeros_,y*torch.sin(x*math.pi/(LENGTH))**2/LENGTH],1),
            32: torch.stack([torch.sin(2*math.pi*x/(3*LENGTH)),zeros_],1),            
            44: torch.stack([(math.pi*y**2/LENGTH**2)*torch.sin(2*math.pi*x/LENGTH),(-2*y/LENGTH)*torch.sin(math.pi*x/LENGTH)**2],1),
            21: torch.stack([torch.square(y)*torch.sin(x*math.pi/LENGTH)/LENGTH**2, torch.square(x)*torch.sin(y*math.pi/LENGTH)/LENGTH**2], 1),
            #3: torch.stack([-x*y*(x-LENGTH)/LENGTH**2,zeros_], 1),
            #5: torch.stack([torch.sin(4*math.pi*x/(3*LENGTH)),zeros_],1),
            #33: torch.stack([(y/LENGTH)*torch.sin(math.pi*x/LENGTH),zeros_],1),
            #6: torch.stack([(2*math.pi*y**2/LENGTH**2)*torch.sin(4*math.pi*x/LENGTH),(-2*y/LENGTH)*torch.sin(2*math.pi*x/LENGTH)**2],1),
            #3: torch.stack([zeros_, y*(torch.square(x)-x*LENGTH)/LENGTH**3], 1),
            #4: torch.stack([zeros_, torch.sin(x*math.pi/LENGTH)*torch.sin(y*math.pi/LENGTH)], 1),
            #20: torch.stack([x*y*(x-LENGTH)/LENGTH**2, y*x*(y-LENGTH)/LENGTH**2], 1),
            #55: torch.stack([zeros_, y*torch.sin(x*math.pi/LENGTH)/LENGTH],1),
            #56: torch.stack([zeros_,np.sin(5*math.pi*y/(3*LENGTH))],1),
            #5: torch.stack([torch.sin(y*math.pi/LENGTH)*torch.sin(x*math.pi/LENGTH), zeros_], 1),
            #6: torch.stack([x*y*(x-LENGTH)/LENGTH**2,zeros_], 1),
            #7: torch.stack([torch.square(x)*(LENGTH-x)*torch.sin(math.pi*y/LENGTH)/LENGTH**3,zeros_], 1),
            # 8: torch.stack([zeros_, (LENGTH**3-x**3)*torch.sin(math.pi*y/LENGTH)/LENGTH**3], 1),
            #9: torch.stack([(x*LENGTH**2-x**3)*torch.sin(y*math.pi/LENGTH)/LENGTH**3, zeros_], 1),
            #10: torch.stack([(x*y*(x-LENGTH)/LENGTH**2)*torch.sin(y*math.pi/LENGTH), zeros_], 1),
            #11: torch.stack([zeros_, (x*y*(y-LENGTH)/LENGTH**2)*torch.sin(x*math.pi/LENGTH)], 1),
            # 12: torch.stack([zeros_, x*y*(y-LENGTH)/LENGTH**2], 1),
            #13: torch.stack([torch.square(y)*torch.sin(x*math.pi/LENGTH)/LENGTH**2, zeros_], 1),
            #14: torch.stack([zeros_, torch.square(x)*torch.sin(y*math.pi/LENGTH)/LENGTH**2], 1),
            #15: torch.stack([(x*y*(x-LENGTH)/LENGTH**2)*torch.sin(x**2*y**2/LENGTH**4), zeros_], 1),
            #16: torch.stack([torch.sin(x*math.pi/LENGTH), zeros_], 1),
            #17: torch.stack([zeros_, torch.sin(y*math.pi/LENGTH)], 1),
            #22: torch.stack([torch.sin(x*math.pi/LENGTH), torch.sin(y*math.pi/LENGTH)], 1),
            #18: torch.stack([torch.sin(x**3*math.pi/LENGTH**3), zeros_], 1),
            # 19: torch.stack([zeros_, torch.sin(y**3*math.pi/LENGTH**3)], 1)
        
        }    

        # Defining virtual strain fields
        virtual_strain = {
            1:torch.stack([ones/LENGTH, zeros, zeros], 1),
            2:torch.stack([zeros,(2*math.pi/(3*LENGTH))*torch.cos(2*cent_y*math.pi/(3*LENGTH)),zeros], 1),
            31: torch.stack([zeros,(torch.sin(cent_x*math.pi/(LENGTH))**2)/LENGTH,(2*math.pi*cent_y)*torch.sin(cent_x*math.pi/LENGTH)*torch.cos(cent_x*math.pi/LENGTH)/LENGTH**2],1),
            32: torch.stack([(2*math.pi/(3*LENGTH))*torch.cos(2*math.pi*cent_x/(3*LENGTH)),zeros,zeros],1),
            44: torch.stack([(2*math.pi**2*cent_y**2/LENGTH**3)*torch.cos(2*math.pi*cent_x/LENGTH),(-2/LENGTH)*torch.sin(math.pi*cent_x/LENGTH)**2, zeros],1),
            21:torch.stack([torch.square(cent_y)*torch.cos(cent_x*math.pi/LENGTH)*math.pi/LENGTH**3, torch.square(cent_x)*torch.cos(cent_y*math.pi/LENGTH)*math.pi/LENGTH**3, (2*cent_y*torch.sin(cent_x*math.pi/LENGTH)/LENGTH**2)+(2*cent_x*torch.sin(cent_y*math.pi/LENGTH)/LENGTH**2)],1),
            #3: torch.stack([cent_y*(-2*cent_x+LENGTH)/LENGTH**2,zeros,-cent_x*(cent_x-LENGTH)/LENGTH**2], 1),
            #5: torch.stack([(4*math.pi/(3*LENGTH))*torch.cos(4*math.pi*cent_x/(3*LENGTH)),zeros,zeros],1),
            #33:torch.stack([(cent_y*math.pi/LENGTH**2)*torch.cos(math.pi*cent_x/LENGTH),zeros,torch.sin(math.pi*cent_x/LENGTH)/LENGTH],1),
            #6: torch.stack([(8*math.pi**2*cent_y/LENGTH**3)*torch.cos(4*math.pi*cent_x/LENGTH),(-2/LENGTH)*np.sin(2*math.pi*cent_x/LENGTH)**2, zeros],1),
            #3:torch.stack([zeros, (torch.square(cent_x)-cent_x*LENGTH)/LENGTH**2, cent_y*(2*cent_x-LENGTH)/LENGTH**3], 1),
            #4:torch.stack([zeros, (math.pi/LENGTH)*torch.sin(cent_x*math.pi/LENGTH)*torch.cos(cent_y*math.pi/LENGTH), (math.pi/LENGTH)*torch.cos(cent_x*math.pi/LENGTH)*torch.sin(cent_y*math.pi/LENGTH)],1),
            #20:torch.stack([cent_y*(2*cent_x-LENGTH)/LENGTH*2, cent_x*(2*cent_y-LENGTH)/LENGTH**2, (cent_x*(2*cent_y-LENGTH)/LENGTH**2) + (cent_y*(cent_y-LENGTH)/LENGTH**2)], 1),
            #55: torch.stack([zeros,torch.sin(cent_x*math.pi/LENGTH)/LENGTH,cent_y*torch.cos(cent_x*math.pi/LENGTH)*math.pi/LENGTH**2],1),
            #56: torch.stack([zeros,torch.cos(5*math.pi*cent_y/(3*LENGTH))*5*math.pi/(3*LENGTH),zeros],1),
            #5:torch.stack([(math.pi/LENGTH)*torch.cos(cent_x*math.pi/LENGTH)*torch.sin(cent_y*math.pi/LENGTH), zeros, (math.pi/LENGTH)*torch.sin(cent_x*math.pi/LENGTH)*torch.cos(cent_y*math.pi/LENGTH)],1),
            #6:torch.stack([(2*cent_x*cent_y-cent_y*LENGTH)/LENGTH**2, zeros, (torch.square(cent_x)-cent_x*LENGTH)/LENGTH**2], 1),
            #7:torch.stack([(2*cent_x*LENGTH-3*torch.square(cent_x))*torch.sin(math.pi*cent_y/LENGTH)/LENGTH**3, zeros, (torch.square(cent_x)*LENGTH-cent_x**3)*torch.cos(math.pi*cent_y/LENGTH)*math.pi/LENGTH**4], axis=1),
            # 8:torch.stack([zeros, (LENGTH**3-cent_x**3)*math.pi*torch.cos(math.pi*cent_y/LENGTH)/LENGTH**4, (-3*torch.square(cent_x)/LENGTH**3)*torch.sin(math.pi*cent_y/LENGTH)], axis=1),
            #9:torch.stack([(LENGTH**2-3*torch.square(cent_x))*torch.sin(math.pi*cent_y/LENGTH)/LENGTH**3, zeros, (cent_x*LENGTH**2-cent_x**3)*torch.cos(math.pi*cent_y/LENGTH)*math.pi/LENGTH**4],1),
            #10:torch.stack([(2*cent_x*cent_y-cent_y*LENGTH)*torch.sin(cent_y*math.pi/LENGTH)/LENGTH**2, zeros, (torch.square(cent_x)-cent_x*LENGTH)*torch.sin(cent_y*math.pi/LENGTH)/LENGTH**2+(cent_x*cent_y*(cent_x-LENGTH))*torch.cos(cent_y*math.pi/LENGTH)*math.pi/LENGTH**3], 1),
            #11:torch.stack([zeros, (2*cent_x*cent_y-cent_x*LENGTH)*torch.sin(cent_x*math.pi/LENGTH)/LENGTH**2, (torch.square(cent_y)-cent_y*LENGTH)*torch.sin(cent_x*math.pi/LENGTH)/LENGTH**2 + (cent_x*cent_y*(cent_y-LENGTH))*torch.cos(cent_x*math.pi/LENGTH)*math.pi/LENGTH**3], 1),
            # 12:torch.stack([zeros, cent_x*(2*cent_y-LENGTH)/LENGTH**2, cent_y*(cent_y-LENGTH)/LENGTH**2], 1),
            #13: torch.stack([torch.square(cent_y)*torch.cos(cent_x*math.pi/LENGTH)*math.pi/LENGTH**3, zeros, 2*cent_y*torch.sin(cent_x*math.pi/LENGTH)/LENGTH**2], 1),
            #14: torch.stack([zeros, torch.square(cent_x)*torch.cos(cent_y*math.pi/LENGTH)*math.pi/LENGTH**3, 2*cent_x*torch.sin(cent_y*math.pi/LENGTH)/LENGTH**2], 1),
            # 15: torch.stack([((2*cent_x*cent_y-cent_y*LENGTH)*torch.sin(torch.square(cent_x)*torch.square(cent_y)/LENGTH**4)/LENGTH**2) + (cent_x*cent_y*(cent_x-LENGTH)*torch.cos(torch.square(cent_x)*torch.square(cent_y)/LENGTH**4)*2*cent_x*torch.square(cent_y)/LENGTH**6), zeros, ((torch.square(cent_x)-cent_x*LENGTH)*torch.sin(torch.square(cent_x)*torch.square(cent_y)/LENGTH**4)/LENGTH**2) + (cent_x*cent_y*(cent_x-LENGTH)*torch.cos(torch.square(cent_x)*torch.square(cent_y)/LENGTH**4)*2*cent_x**2*cent_y/LENGTH**6)],1),
            # 16: torch.stack([torch.cos(cent_x*math.pi/LENGTH)*math.pi/LENGTH, zeros, zeros],1),
            # 17: torch.stack([zeros, torch.cos(cent_y*math.pi/LENGTH)*math.pi/LENGTH, zeros],1),
            #22: torch.stack([torch.cos(cent_x*math.pi/LENGTH)*math.pi/LENGTH, torch.cos(cent_y*math.pi/LENGTH)*math.pi/LENGTH, zeros], 1),
            #18: torch.stack([torch.cos(cent_x**3*math.pi/LENGTH**3)*3*torch.square(cent_x)*math.pi/LENGTH**3, zeros, zeros], 1),
            # 19: torch.stack([zeros, torch.cos(cent_y**3*math.pi/LENGTH**3)*3*torch.square(cent_y)*math.pi/LENGTH**3, zeros], 1)
            
        }
    elif dir == 2:
         # Virtual displacement fields
        virtual_disp = {
            1: torch.stack([zeros_, y/LENGTH],1),
            2: torch.stack([zeros_,torch.sin(2*y*math.pi/(3*LENGTH))],1),
            
            2: torch.stack([zeros_,torch.sin(2*y*math.pi/(3*LENGTH))],1),
            3: torch.stack([x*torch.sin(y*math.pi/(LENGTH))**2/LENGTH,zeros_],1),
            5: torch.stack([(-x/LENGTH)*torch.sin(math.pi*y/LENGTH)**2,(2*math.pi*x/LENGTH**2)*torch.sin(2*math.pi*y/LENGTH)],1),
        
        }    

        # Defining virtual strain fields
        virtual_strain = {
            1:torch.stack([zeros, ones/LENGTH, zeros], 1),
            2:torch.stack([zeros,2*math.pi*torch.cos(2*cent_y*math.pi/(3*LENGTH))/(3*LENGTH),zeros], 1),
            3: torch.stack([torch.sin(cent_y*math.pi/(LENGTH))**2/LENGTH,zeros,cent_x*2*torch.sin(cent_y*math.pi/(LENGTH))*math.pi/LENGTH**2],1),
            5: torch.stack([(-1/LENGTH)*torch.sin(math.pi*cent_y/LENGTH)**2, (4*math.pi**2*cent_y/LENGTH**3)*torch.cos(2*math.pi*cent_y/LENGTH),zeros],1),
            
        }


    # Total number of virtual fields
    total_vfs = len(virtual_disp.keys())

    # Converting virtual displacement/strain fields dictionaries into a tensors
    v_disp = torch.stack(list(virtual_disp.values()))
    v_strain = torch.stack(list(virtual_strain.values()))

    return total_vfs, v_disp, v_strain

def standardize_data(X, y, scaler_x = None, scaler_y = None):

        if scaler_x == None: 
            scaler_x = preprocessing.StandardScaler()
            scaler_x.fit(X)
            X = scaler_x.transform(X)
        else:
            X = scaler_x.transform(X)

        if scaler_y == None:
            scaler_y = preprocessing.StandardScaler()
            scaler_y.fit(y)
            y = scaler_y.transform(y)
        else:
            y = scaler_y.transform(y)

        return X, y, scaler_x, scaler_y

def next_batch(inputs, targets, batchSize):
	# loop over the dataset
	for i in range(0, inputs.shape[0], batchSize):
		# yield a tuple of the current batched data and labels
		yield (inputs[i:i + batchSize], targets[i:i + batchSize])

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
    num_batches = len(dataloader)
    losses = torch.zeros(num_batches)
    

    for batch in range(num_batches):
        # Extracting variables for training
        X_train, y_train = dataloader[batch]
        
        # Converting to pytorch tensors
        X_train = torch.tensor(X_train, dtype=torch.float64)
        y_train = torch.tensor(y_train, dtype=torch.float64)
         
        pred=model(X_train)

        # Computing losses        
        loss = f_loss(pred,y_train)
        
        # Backpropagation and weight's update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Saving loss values
        losses[batch] = loss
        
        print('\r>Train: %d/%d' % (batch + 1, num_batches), end='')

    return losses

def test_loop(dataloader, model, loss_fn):
    
    num_batches = len(dataloader)
    test_losses = torch.zeros(num_batches)
    
    model.eval()
    with torch.no_grad():

        for batch in range(num_batches):

            # Extracting variables for testing
            X_test, y_test = dataloader[batch]
            
            # Converting to pytorch tensors
            X_test = torch.tensor(train_generator.scaler_x.transform(X_test), dtype=torch.float64)
            
            y_test = torch.tensor(train_generator.scaler_y.transform(y_test), dtype=torch.float64)
            
            pred = model(X_test)

            # Computing losses        
            test_loss = f_loss(pred,y_test)
            test_losses[batch] = test_loss

            print('\r>Test: %d/%d' % (batch + 1, num_batches), end='')

    return test_losses

# -------------------------------
#           Main script
# -------------------------------

torch.set_default_dtype(torch.float64)

# Specifying random seed
random.seed(SEED)

# Loading data
df_list, _ = load_dataframes(TRAIN_MULTI_DIR)

# Sampling data pass random seed for random sampling
sampled_dfs = data_sampling(df_list, DATA_SAMPLES)

# Merging training data
data = pd.concat(sampled_dfs, axis=0, ignore_index=True)

BATCH_SIZE = 32

# Selecting model features
X, y, f, coord = select_features_multi(data)


# from scipy.signal import savgol_filter
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.optimize import curve_fit
# from scipy.interpolate import interp1d
# from scipy.signal import savgol_filter

# indices = np.arange(0,len(X)+DATA_SAMPLES-1,DATA_SAMPLES)

# for i in range(len(indices)):
#     curr = indices[i]
#     next = indices[i+1]
#     x = X['eyy_t'].iloc[curr:next]
#     y = Y['syy_t'].iloc[curr:next]

#     xx = np.linspace(x.min(),x.max(), DATA_SAMPLES)
#     # interpolate + smooth
#     itp = interp1d(x,y, kind='linear')
#     #window_size, poly_order = 5, 1
#     #yy_sg = savgol_filter(itp(xx), window_size, poly_order)

#     fig, ax = plt.subplots(figsize=(7, 4))
#     ax.plot(x, y, 'r.', label= 'Unsmoothed curve')
#     ax.plot(xx, itp(xx), 'k', label= "Smoothed curve")
#     plt.legend(loc='best')
#     plt.grid()
#     plt.show()



# Performing test/train split
partition = {"train": None, "test": None}

# dataset_size = len(data)
# indices = list(range(dataset_size))
# split = int(np.floor(TEST_SIZE * dataset_size))
# shuffle_dataset = True
# if shuffle_dataset :
#     np.random.seed(SEED)
#     np.random.shuffle(indices)
# partition['train'], partition['test'] = indices[split:], indices[:split]


(trainX, testX, trainY, testY) = train_test_split(X, y,	test_size=TEST_SIZE, random_state=SEED)

trainX, trainY, scaler_x, scaler_y = standardize_data(trainX,trainY)
testX, testY, _, _ = standardize_data(testX, testY, scaler_x, scaler_y)

trainX = torch.from_numpy(trainX)
testX = torch.from_numpy(testX)
trainY = torch.from_numpy(trainY)
testY = torch.from_numpy(testY)


# Model variables
N_INPUTS = trainX.shape[1]
N_OUTPUTS = trainY.shape[1]

N_UNITS = 50
H_LAYERS = 3

model_1 = NeuralNetwork(N_INPUTS, N_OUTPUTS, N_UNITS, H_LAYERS)

model_1.apply(init_weights)

# Training variables
EPOCHS = 500

# Optimization variables
learning_rate = 0.01
loss_fn = custom_loss
f_loss = torch.nn.MSELoss()


optimizer = torch.optim.Adam(params=list(model_1.parameters()), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=12, factor=0.2, threshold=1e-3, min_lr=1e-5)

# create a template to summarize current training progress
trainTemplate = "epoch: {} test loss: {:.3e} test accuracy: {:.3e}"

epochs_ = []
train_loss = []
val_loss = []
# loop through the epochs
for epoch in range(0, EPOCHS):
    # initialize tracker variables and set our model to trainable
    print("[INFO] epoch: {}...".format(epoch + 1))
    trainLoss = 0
    samples = 0
    epochs_.append(epoch+1)
    model_1.train()

    # loop over the current batch of data
    for (batchX, batchY) in next_batch(trainX, trainY, BATCH_SIZE):
        # flash data to the current device, run it through our
        # model, and calculate loss
        predictions = model_1(batchX)
        loss = f_loss(predictions, batchY)

        # zero the gradients accumulated from the previous steps,
        # perform backpropagation, and update model parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update training loss, accuracy, and the number of samples
        # visited
        trainLoss += loss.item()
        samples += 1

    # display model progress on the current training batch
    trainTemplate = "epoch: {} train loss: {:.3e}"
    print(trainTemplate.format(epoch + 1, (trainLoss / samples)))
    train_loss.append(trainLoss / samples)
    scheduler.step(train_loss[epoch])
    # initialize tracker variables for testing, then set our model to
    # evaluation mode
    testLoss = 0
    samples = 0
    model_1.eval()

    # initialize a no-gradient context
    with torch.no_grad():
        # loop over the current batch of test data
        for (batchX, batchY) in next_batch(testX, testY, BATCH_SIZE):
        # flash the data to the current device

            # run data through our model and calculate loss
            predictions = model_1(batchX)
            loss = f_loss(predictions, batchY)

            # update test loss, accuracy, and the number of
            # samples visited
            testLoss += loss.item()
            samples += 1

    # display model progress on the current test batch
    testTemplate = "epoch: {} test loss: {:.3e}"
    print(testTemplate.format(epoch + 1, (testLoss / samples)))
    val_loss.append(testLoss / samples)
    print("")

    

# # Container variables for history purposes
# train_loss = []
# v_work = []
# val_loss = []
# epochs_ = []
# l = []
# # Initializing the early_stopping object
# #early_stopping = EarlyStopping(patience=12, verbose=True)

# for t in range(epochs):

#     print('\r--------------------\nEpoch [%d/%d]' % (t + 1, epochs))

#     epochs_.append(t+1)
    
#     start_epoch = time.time()

#     #Shuffling batches
#     train_generator.on_epoch_end()
    
#     # Train loop
#     start_train = time.time()
#     batch_losses = train_loop(train_generator, model_1, loss_fn, optimizer)
    
#     train_loss.append(torch.mean(batch_losses).item())

#     end_train = time.time()
    
#     #Apply learning rate scheduling if defined
#     try:
#         scheduler.step(train_loss[t])       
#         print('. t_loss: %.3e -- %.3fs' % (train_loss[t], end_train - start_train))
#     except:
#         print('. loss: %.3e -- %.3fs' % (train_loss[t], end_train - start_train))

#     # Test loop
#     start_test = time.time()

#     batch_val_losses = test_loop(test_generator, model_1, loss_fn)

#     val_loss.append(torch.mean(batch_val_losses).item())

#     end_test = time.time()

#     print('. v_loss: %.3e -- %.3fs' % (val_loss[t], end_test - start_test))

#     end_epoch = time.time()

#     # # Check validation loss for early stopping
#     # early_stopping(val_loss[t], model)

#     # if early_stopping.early_stop:
#     #      print("Early stopping")
#     #      break

print("Done!")

# load the last checkpoint with the best model
#model.load_state_dict(torch.load('checkpoint.pt'))

epochs_ = np.reshape(np.array(epochs_), (len(epochs_),1))
train_loss = np.reshape(np.array(train_loss), (len(train_loss),1))
val_loss = np.reshape(np.array(val_loss), (len(val_loss),1))

history = pd.DataFrame(np.concatenate([epochs_, train_loss, val_loss], axis=1), columns=['epoch','loss','val_loss'])

task = r'[%i-%ix%i-%i]-%s' % (N_INPUTS, N_UNITS, H_LAYERS, N_OUTPUTS, TRAIN_MULTI_DIR.split('/')[-2])

import os
output_task = 'outputs/' + TRAIN_MULTI_DIR.split('/')[-2] + '_direct'
output_loss = 'outputs/' + TRAIN_MULTI_DIR.split('/')[-2] + '_direct/loss/'
output_prints = 'outputs/' + TRAIN_MULTI_DIR.split('/')[-2] + '_direct/prints/'
output_models = 'outputs/' + TRAIN_MULTI_DIR.split('/')[-2] + '_direct/models/'
output_val = 'outputs/' + TRAIN_MULTI_DIR.split('/')[-2] + '_direct/val/'

directories = [output_task, output_loss, output_prints, output_models, output_val]

for dir in directories:
    try:
        os.makedirs(dir)  
        
    except FileExistsError:
        pass

history.to_csv(output_loss + task + '.csv', sep=',', encoding='utf-8', header='true')

plot_history(history, output_prints, True, task)

torch.save(model_1.state_dict(), output_models + task + '_1.pt')

joblib.dump(scaler_x, output_models + task + '-scaler_x.pkl')
joblib.dump(scaler_y, output_models + task + '-scaler_y.pkl')
#joblib.dump(stress_scaler, output_models + task + '-scaler_stress.pkl')
# ---------------------------------
#    Library and function imports
# ---------------------------------
import re
import operator
from turtle import color
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

import tensorflow as tf
import pandas as pd

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
from sympy import pi
import geotorch
from torch.nn.utils import (
  parameters_to_vector as Params2Vec,
  vector_to_parameters as Vec2Params
)
import loss_landscapes
import loss_landscapes.metrics

from abc import ABC, abstractmethod
from loss_landscapes.model_interface.model_wrapper import ModelWrapper
from mayavi import mlab

import matplotlib.pyplot as plt


# -------------------------------
#        Class definitions
# -------------------------------
class Metric(ABC):
    """ A quantity that can be computed given a model or an agent. """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def __call__(self, model_wrapper: ModelWrapper):
        pass

class Loss(Metric):
    """ Computes a specified loss function over specified input-output pairs. """
    def __init__(self, loss_fn, X, f, coord):
        super().__init__()
        self.loss_fn = loss_fn
        self.X = torch.from_numpy(X)
        self.f = torch.from_numpy(f)
        self.coord = torch.from_numpy(coord)

    def __call__(self, model_wrapper: ModelWrapper) -> float:

        t_pts = train_generator.t_pts
        n_elems = batch_size//t_pts

        # Extracting centroid coordinates
        dir = self.coord[:,0][0]
        id = self.coord[:,1]
        cent_x = self.coord[:,2]
        cent_y = self.coord[:,3]
        area = torch.reshape(self.coord[:,4],[batch_size,1])

        # Defining surface coordinates where load is applied
        n_surf_nodes = int((9**0.5) + 1)
    
        x = LENGTH * torch.ones(n_surf_nodes, dtype=torch.float)
        y = torch.tensor(np.linspace(0, LENGTH, n_surf_nodes), dtype=torch.float)

        total_vfs, v_disp, v_strain = get_v_fields(cent_x, cent_y, x, y, dir)

        pred=model_wrapper.forward(self.X)

        int_work = torch.reshape((pred * v_strain * area * ELEM_THICK),[total_vfs,t_pts,n_elems,3])
        
        # Extracting global force components and filtering duplicate values
    
        f = self.f[:,:2][::n_elems,:]

        # Computing external virtual work
        
        ext_work = torch.sum(torch.reshape(f,[t_pts,1,2])*torch.mean(v_disp,1),-1,keepdim=True).permute(1,0,2)

        return self.loss_fn(int_work, ext_work).item()


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, deformation, stress, force, coord, list_IDs, batch_size, shuffle, std=True, t_pts=1):
        super().__init__()
        self.X = deformation
        self.y = stress
        self.f = force
        self.coord = coord[['dir','id','x','y','area']]
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
        # Generate indexes of the batch
        indexes = np.array([self.indexes[index]+i for i in range(self.batch_size)])

        # Generate data
        if self.shuffle == True:
            if batch_size > 9:
                index_groups = np.array_split(indexes, self.t_pts)
                permuted = [np.random.permutation(index_groups[i]) for i in range(len(index_groups))]
                indexes = np.hstack(permuted)
                X, y, f, coord = self.__data_generation(indexes)
            else:
                X, y, f, coord = self.__data_generation(np.random.permutation(indexes))
        else:
            X, y, f, coord = self.__data_generation(indexes)
        
        return X, y, f, coord

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(0, len(self.list_IDs), self.batch_size)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def standardize(self):
        idx = self.X.index
        self.X, _, _, self.scaler_x, _, _ = standardize_data(self.X, self.y, self.f)
        #self.f, self.scaler_f = standardize_(self.f)
        #self.coord, self.scaler_coord = standardize_(self.coord[['x','y']])

        self.X = pd.DataFrame(self.X, index=idx)
        #self.y = pd.DataFrame(self.y, index=idx)
        #self.f = pd.DataFrame(self.f, index=idx)
        #self.coord = pd.DataFrame(self.coord, index=idx)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        X = np.asarray(self.X.iloc[list_IDs_temp], dtype=np.float32)
        y = np.asarray(self.y.iloc[list_IDs_temp], dtype=np.float32)
        f = np.asarray(self.f.iloc[list_IDs_temp], dtype=np.float32)
        coord = np.asarray(self.coord.iloc[list_IDs_temp], dtype=np.float32)
        return X, y, f, coord

class NeuralNetwork(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_hidden_layers=1):
        super(NeuralNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size  = hidden_size
        self.n_hidden_layers = n_hidden_layers
        self.output_size = output_size

        self.layers = torch.nn.ModuleList()

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


# -------------------------------
#       Method definitions
# -------------------------------

def batch_jacobian(f, x):
    f_sum = lambda x: torch.sum(f(x), axis=0)
    return jacobian(f_sum, x,create_graph=True).permute(1,0,2)

def get_v_fields(cent_x, cent_y, x, y, dir):

    # Auxialiary vectors
    zeros_ = torch.zeros_like(x, dtype=torch.float32)

    ones = torch.ones(size=(batch_size,), dtype=torch.float32)
    zeros = torch.zeros(size=(batch_size,), dtype=torch.float32)

    pi_ = math.pi

    eps = np.finfo(np.float32).eps

    if dir == 1:
        # Virtual displacement fields
        virtual_disp = {
            1: torch.stack([x/LENGTH, zeros_], 1),
           
            #31: torch.stack([zeros_,y*torch.sin(x*math.pi/(LENGTH))**2/LENGTH],1),
            32: torch.stack([torch.sin(2*pi_*x/(3*LENGTH)),zeros_],1),            
            44: torch.stack([(pi_*(y**2/LENGTH**2))*torch.sin(2*pi_*x/LENGTH),(-2*y/LENGTH**2)*torch.sin(pi_*x/LENGTH)**2],1),
            21: torch.stack([(y/LENGTH)**2*torch.sin(x*pi_/LENGTH), (x/LENGTH)**2*torch.sin(y*pi_/LENGTH)], 1),
            
        
        }    

        # Defining virtual strain fields
        virtual_strain = {
            1:torch.stack([ones/LENGTH, zeros, zeros], 1),
            
            #31: torch.stack([zeros,(torch.sin(cent_x*math.pi/(LENGTH))**2)/LENGTH,(2*math.pi*cent_y)*torch.sin(cent_x*math.pi/LENGTH)*torch.cos(cent_x*math.pi/LENGTH)/LENGTH**2],1),
            32: torch.stack([(2*math.pi/(3*LENGTH))*torch.cos(2*math.pi*cent_x/(3*LENGTH)),zeros,zeros],1),
            44: torch.stack([(2*math.pi**2*cent_y**2/LENGTH**3)*torch.cos(2*math.pi*cent_x/LENGTH),(-2/LENGTH**2)*torch.sin(math.pi*cent_x/LENGTH)**2, zeros],1),
            21:torch.stack([torch.square(cent_y)*torch.cos(cent_x*math.pi/LENGTH)*math.pi/LENGTH**3, torch.square(cent_x)*torch.cos(cent_y*math.pi/LENGTH)*math.pi/LENGTH**3, 0.5*((2*cent_y*torch.sin(cent_x*math.pi/LENGTH)/LENGTH**2)+(2*cent_x*torch.sin(cent_y*math.pi/LENGTH)/LENGTH**2))],1),
            
            
        }
    
    # Total number of virtual fields
    total_vfs = len(virtual_disp.keys())

    # Converting virtual displacement/strain fields dictionaries into a tensors
    v_disp = torch.stack(list(virtual_disp.values()))
    v_strain = torch.stack(list(virtual_strain.values()))

    v_disp[torch.abs(v_disp) < 1e-9] = 0
    v_strain[torch.abs(v_strain) < 1e-9] = 0
    return total_vfs, v_disp, v_strain


def tau_2d(alpha, beta, theta_ast):
  a = alpha * theta_ast[:,None,None]
  b = beta * alpha * theta_ast[:,None,None]
  return a + b


torch.set_default_dtype(torch.float32)
torch.set_printoptions(precision=8)

# Specifying random seed
random.seed(SEED)

# Loading data
df_list, _ = load_dataframes(TRAIN_MULTI_DIR)

# Sampling data pass random seed for random sampling
sampled_dfs = data_sampling(df_list, DATA_SAMPLES)

# Merging training data
data = pd.concat(sampled_dfs, axis=0, ignore_index=True)

t_pts = 8

# Performing test/train split
partition = {"train": None, "test": None}

if t_pts == DATA_SAMPLES:
    # Reorganizing dataset by tag, subsequent grouping by time increment
    data_by_tag = [df for _, df in data.groupby(['tag'])]
    random.shuffle(data_by_tag)
    data_by_t = [[df for _, df in group.groupby(['t'])] for group in data_by_tag]
    #random.shuffle(data_by_t)
    data_by_batches = list(itertools.chain(*data_by_t))
    #random.shuffle(data_by_batches)

    data = pd.concat(data_by_batches).reset_index(drop=True)

    partition['train'] = data.index.tolist()
    
else:
    # Reorganizing dataset by time increment, subsequent grouping by tag and final shuffling
    data_by_t = [df for _, df in data.groupby(['t'])]
    random.shuffle(data_by_t)
    data_by_tag = [[df for _, df in group.groupby(['tag'])] for group in data_by_t]
    random.shuffle(data_by_tag)
    data_by_batches = list(itertools.chain(*data_by_tag))
    random.shuffle(data_by_batches)

    data = pd.concat(data_by_batches).reset_index(drop=True)

    partition['train'] = data.index.tolist()

batch_size = len(data_by_batches[0]) * t_pts

x_scaler = joblib.load('outputs/9-elem-1000-elastic_indirect/models/[3-6x1-3]-9-elem-1000-elastic-4-VFs-scaler_x.pkl')

# Selecting model features
X, y, f, coord = select_features_multi(data)

X = pd.DataFrame(x_scaler.transform(X),index=X.index.tolist())

# Preparing data generators for mini-batch training
train_generator = DataGenerator(X, y, f, coord, partition["train"], batch_size, True, std=True, t_pts=t_pts)

# Model variables
N_INPUTS = X.shape[1]
N_OUTPUTS = y.shape[1]

N_UNITS = 6
H_LAYERS = 1

model_1 = NeuralNetwork(N_INPUTS, N_OUTPUTS, N_UNITS, H_LAYERS)

model_1.load_state_dict(torch.load('outputs/9-elem-1000-elastic_indirect/models/[3-6x1-3]-9-elem-1000-elastic-4-VFs_1.pt'))

X,y,f,coord = iter(train_generator).__next__()

criterion = custom_loss
metric = Loss(criterion,X,f,coord)

STEPS = 1000

loss_data_fin = loss_landscapes.random_plane(model_1, metric, 10, STEPS, normalization='filter', deepcopy_model=True)

fig, ax = plt.subplots()
cs=ax.contour(loss_data_fin, levels=50)
ax.clabel(cs,inline=1, fontsize=6)
plt.title('Loss Contours around Trained Model')
plt.show()

joblib.dump(loss_data_fin,'loss_data_fin.pkl')

X,Y=np.mgrid[-STEPS*0.5:STEPS*0.5,-STEPS*0.5:STEPS*0.5]

mlab.figure(bgcolor=(1,1,1),fgcolor=(0,0,0))
surf = mlab.surf(X,Y,loss_data_fin,warp_scale='auto')
mlab.orientation_axes()
#mlab.axes(color=(0.2,0.2,0.2))
#mlab.colorbar(orientation='vertical')
#mlab.outline(color=(0.2,0.2,0.2))
mlab.show()
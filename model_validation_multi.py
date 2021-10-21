from tensorflow.python.keras.backend import dtype
import constants
import joblib
import tensorflow as tf
from tensorflow import keras
from functions import custom_loss, load_dataframes, data_sampling, select_features, select_features_multi, standardize_data
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from cycler import cycler
from operator import itemgetter
import torch.nn.functional as F

import torch
from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_hidden_layers=1):
        super(NeuralNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size  = hidden_size
        self.n_hidden_layers = n_hidden_layers
        self.output_size = output_size

        self.layers = nn.ModuleList()
        self.layers.append(torch.nn.Linear(self.input_size, self.hidden_size))

        for i in range(self.n_hidden_layers):
            self.layers.append(nn.Linear(self.hidden_size, self.hidden_size))

        self.layers.append(torch.nn.Linear(self.hidden_size, self.output_size))

        self.activation = torch.nn.PReLU(self.hidden_size)
        
        # self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        # self.relu1 = torch.nn.RReLU()
        # self.fc2 = torch.nn.Linear(self.hidden_size,self.hidden_size)
        # self.relu2 = torch.nn.RReLU()
        # self.fc3 = torch.nn.Linear(self.hidden_size, 3)

    def forward(self, x):

        x = self.layers[0](x)
        for layer in self.layers[1:-1]:
            x =self.activation(layer(x))
        # hidden1 = self.fc1(x)
        # relu1 = self.relu1(hidden1)
        # hidden2 = self.fc2(relu1)
        # relu2 = self.relu2(hidden2)
        return self.layers[-1](x)

plt.rcParams.update(constants.PARAMS)

default_cycler = (cycler(color=["#ef476f","#118ab2","#073b4c"]))

plt.rc('axes', prop_cycle=default_cycler)

# Loading data
df_list, file_names = load_dataframes(constants.TRAIN_MULTI_DIR)

file_names = [file_name.split('/')[-1] for file_name in file_names]
# Loading data scalers
#x_scaler, y_scaler = joblib.load('models/ann3/scalers.pkl')

# Loading ANN model
model = NeuralNetwork(6, 3, 32, 1)
model.load_state_dict(torch.load('models/ann_torch/model_1'))
model.eval()
#model = keras.models.load_model('models/ann3', compile=False)

#model.summary()

# Sampling data pass random seed for random sampling
sampled_dfs = data_sampling(df_list, constants.DATA_SAMPLES)

with torch.no_grad():
    for i, df in enumerate(sampled_dfs):

        X, y, _, _ = select_features_multi(df)

        # Apply previous validation dataset
        #X_val, y_val, _, _ = standardize_data(X, y, x_scaler, y_scaler)

        #y_pred = model.predict(X_val)

        # y_pred_inv = model.predict(X)
        y_pred_inv = model(torch.tensor(X.values).float()).detach().numpy()

        #y_pred_inv = y_scaler.inverse_transform(y_pred)

        ex_var_abaqus = df['exx_t']
        ey_var_abaqus = df['eyy_t']
        exy_var_abaqus = df['exy_t']
        sx_var_abaqus = df['sxx_t']
        sy_var_abaqus = df['syy_t']
        sxy_var_abaqus = df['sxy_t']

        sx_pred_var = y_pred_inv[:,0]
        sy_pred_var = y_pred_inv[:,1]
        sxy_pred_var = y_pred_inv[:,2]

        # sx_pred_var = y_pred[:,0]
        # sy_pred_var = y_pred[:,1]
        # sxy_pred_var = y_pred[:,2]
        
        fig , (ax1, ax2, ax3) = plt.subplots(1,3)
        fig.suptitle(r''+ df['tag'][0].replace('_','\_') + ': element \#' + str(df['id'][0]),fontsize=14)
    
        ax1.plot(ex_var_abaqus, sx_var_abaqus, label='ABAQUS')
        ax1.plot(ex_var_abaqus, sx_pred_var, label='ANN')
        ax1.set(xlabel=r'$\varepsilon$', ylabel=r'$\sigma_{xx}$ [MPa]')
        ax2.plot(ey_var_abaqus, sy_var_abaqus, label='ABAQUS')
        ax2.plot(ey_var_abaqus, sy_pred_var, label='ANN')
        ax2.set(xlabel=r'$\varepsilon$', ylabel=r'$\sigma_{yy}$ [MPa]')
        ax3.plot(exy_var_abaqus, sxy_var_abaqus, label='ABAQUS')
        ax3.plot(exy_var_abaqus, sxy_pred_var, label='ANN')
        ax3.set(xlabel=r'$\varepsilon$', ylabel=r'$\tau_{xy}$ [MPa]')
        handles, labels = ax3.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center')

        #results = model.evaluate(X_val,y_val[:,:3])
        
    plt.show()
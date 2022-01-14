from tensorflow.python.keras.backend import dtype
import constants
import joblib
import tensorflow as tf
from tensorflow import keras
from functions import custom_loss, load_dataframes, data_sampling, select_features_multi, standardize_data
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from cycler import cycler
from operator import itemgetter
import torch.nn.functional as F
import torchmetrics
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
        #self.layers.append(torch.nn.Linear(self.input_size, self.hidden_size, bias=True))

        for i in range(self.n_hidden_layers):
            if i == 0:
                in_ = self.input_size
            else:
                in_ = self.hidden_size

            self.layers.append(torch.nn.Linear(in_, self.hidden_size, bias=True))

        self.layers.append(torch.nn.Linear(self.hidden_size, self.output_size, bias=True))

        self.activation = torch.nn.PReLU(self.hidden_size)
        #self.activation = torch.nn.Tanh()

    def forward(self, x):

        #x = self.layers[0](x)
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
            
        return self.layers[-1](x)

plt.rcParams.update(constants.PARAMS)

default_cycler = (cycler(color=["#ef476f","#118ab2","#073b4c"]))

plt.rc('axes', prop_cycle=default_cycler)

torch.set_default_dtype(torch.float64)

# Loading data
df_list, file_names = load_dataframes(constants.TRAIN_MULTI_DIR)

file_names = [file_name.split('/')[-1] for file_name in file_names]
# Loading data scalers
x_scaler = joblib.load('outputs/9-elem-200-plastic_feedback/models/[6-8x1-3]-9-elem-200-plastic-6-VFs-scaler_x.pkl')
s_scaler = joblib.load('outputs/9-elem-200-plastic_feedback/models/[6-8x1-3]-9-elem-200-plastic-6-VFs-scaler_stress.pkl')

# Loading ANN model
model_1 = NeuralNetwork(9, 3, 8, 1)

model_1.load_state_dict(torch.load('outputs/9-elem-200-plastic_feedback/models/[6-8x1-3]-9-elem-200-plastic-6-VFs_1.pt'))

model_1.eval()

# Sampling data pass random seed for random sampling
sampled_dfs = data_sampling(df_list, constants.DATA_SAMPLES)

#mean_squared_error = torchmetrics.MeanSquaredLogError()
mean_squared_error = torchmetrics.R2Score()

print("----------------------\nMSE\n----------------------\n\tSxx\tSyy\tSxy\n")
with torch.no_grad():
    for i, df in enumerate(sampled_dfs):

        s_hist = np.zeros((len(df),3))

        X, y, _, _ = select_features_multi(df)
        X_scaled=torch.tensor(x_scaler.transform(X))
        
        for k in range(len(df)):

            s_scaled = torch.tensor(s_scaler.transform(s_hist))

            if k==0:
                input = torch.reshape(torch.cat([torch.tensor(s_hist[k,:]),X_scaled[k,:]],0), [1,9])
                y_pred_inv = model_1(input)
            else:
                input = torch.reshape(torch.cat([torch.tensor(s_hist[k-1,:]),X_scaled[k,:]],0), [1,9])
                y_pred_inv = model_1(input)
                
            s_hist[k] = y_pred_inv.detach().numpy()

        #y_pred_inv_1 = model_1(torch.tensor(X_scaled[:,[0,3,6]]))
        # y_pred_inv_2 = model_2(torch.tensor(X_scaled[:,[1,4,7]]))
        # y_pred_inv_3 = model_3(torch.tensor(X_scaled[:,[2,5,8]]))

        #y_pred_inv = y_scaler.inverse_transform(y_pred)

        ex_var_abaqus = df['exx_t']
        ey_var_abaqus = df['eyy_t']
        exy_var_abaqus = df['exy_t']
        sx_var_abaqus = df['sxx_t']
        sy_var_abaqus = df['syy_t']
        sxy_var_abaqus = df['sxy_t']

        sx_pred_var = s_hist[:,0]
        sy_pred_var = s_hist[:,1]
        sxy_pred_var = s_hist[:,2]

        mse_x = mean_squared_error(sx_pred_var, torch.tensor(sx_var_abaqus))
        mse_y = mean_squared_error(sy_pred_var, torch.tensor(sy_var_abaqus))
        mse_xy = mean_squared_error(sxy_pred_var, torch.tensor(sxy_var_abaqus))

        print("%i\t%0.5f\t%0.5f\t%0.5f" % (i, mse_x, mse_y, mse_xy))
        
        fig , (ax1, ax2, ax3) = plt.subplots(1,3)
        fig.suptitle(r''+ df['tag'][0].replace('_','\_') + ': element \#' + str(df['id'][0]),fontsize=14)
        fig.set_size_inches(10, 5)
        fig.subplots_adjust(bottom=0.2, top=0.8)
    
        ax1.plot(ex_var_abaqus, sx_var_abaqus, label='ABAQUS')
        ax1.plot(ex_var_abaqus, sx_pred_var, label='ANN')
        ax1.set(xlabel=r'$\varepsilon_{xx}$', ylabel=r'$\sigma_{xx}$ [MPa]')
        #ax1.set_title(r'$\text{MSE}=%0.3f$' % (mse_x), fontsize=11)
        ax2.plot(ey_var_abaqus, sy_var_abaqus, label='ABAQUS')
        ax2.plot(ey_var_abaqus, sy_pred_var, label='ANN')
        ax2.set(xlabel=r'$\varepsilon_{yy}$', ylabel=r'$\sigma_{yy}$ [MPa]')
        ax2.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
        #ax2.set_title(r'$\text{MSE}=%0.3f$' % (mse_y), fontsize=11)
        ax3.plot(exy_var_abaqus, sxy_var_abaqus, label='ABAQUS')
        # ax3.plot(exy_var_abaqus, func(exy_var_abaqus, *popt), label='ABAQUS')
        ax3.plot(exy_var_abaqus, sxy_pred_var, label='ANN')
        ax3.set(xlabel=r'$\varepsilon_{xy}$', ylabel=r'$\tau_{xy}$ [MPa]')
        ax3.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
        #ax3.set_title(r'$\text{MSE}=%0.3f$' % (mse_xy), fontsize=11)
        handles, labels = ax3.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center',ncol=2)
        
        plt.show()

        # if df['id'][0]==9 or df['id'][0]==1:
        #     predictions = pd.DataFrame(y_pred_inv, columns=['pred_x','pred_y','pred_xy'])
        #     results = pd.concat([df[['exx_t','eyy_t','exy_t','sxx_t','syy_t','sxy_t']],predictions], axis=1)
        #     results.to_csv('outputs/9-elem-200-plastic_feedback/val/' + df['tag'][0]+'_'+str(df['id'][0])+'.csv', header=True, sep=',',float_format='%.8f')
        #     print('hey')

        
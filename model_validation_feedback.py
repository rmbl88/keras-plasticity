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
import itertools

import torch

from torch import nn
from torchmetrics import MeanSquaredError
from scipy import fftpack
from scipy.signal import find_peaks

def despike(yi, th=1.e-8):
    '''Remove spike from array yi, the spike area is where the difference between 
    the neigboring points is higher than th.'''
    y = np.copy(yi) # use y = y1 if it is OK to modify input array
    n = len(y)
    x = np.arange(n)
    c = np.argmax(y)
    d = abs(np.diff(y))
    try:
        l = c - 1 - np.where(d[c-1::-1]<th)[0][0]
        r = c + np.where(d[c:]<th)[0][0] + 1
    except: # no spike, return unaltered array
        return y
    # for fit, use area twice wider then the spike
    if (r-l) <= 3:
        l -= 1
        r += 1
    s = int(round((r-l)/2.))
    lx = l - s
    rx = r + s
    # make a gap at spike area
    xgapped = np.concatenate((x[lx:l],x[r:rx]))
    ygapped = np.concatenate((y[lx:l],y[r:rx]))
    # quadratic fit of the gapped array
    z = np.polyfit(xgapped,ygapped,2)
    p = np.poly1d(z)
    y[l:r] = p(x[l:r])
    return y

from scipy.signal import savgol_filter

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

        #self.activation = torch.nn.PReLU(self.hidden_size)
        self.activation = torch.nn.Tanh()

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
#x_scaler = joblib.load('outputs/9-elem-200-elastic_testfull/models/[6-4x1-3]-9-elem-200-elastic-4-VFs-scaler_x.pkl')
#x_scaler = joblib.load('outputs/9-elem-200-plastic_testfull/models/[6-8x1-3]-9-elem-200-plastic-6-VFs-scaler_x.pkl')
x_scaler = joblib.load('outputs/9-elem-200-plastic_feedback/models/[27-50x1-27]-9-elem-200-plastic-6-VFs-scaler_x.pkl')


# Loading ANN model
model_1 = NeuralNetwork(27, 27, 50, 1)
# model_2 = NeuralNetwork(3, 1, 8, 1)
# model_3 = NeuralNetwork(3, 1, 8, 1)
#model_1.load_state_dict(torch.load('outputs/9-elem-200-elastic_testfull/models/[6-4x1-3]-9-elem-200-elastic-4-VFs.pt'))
#model_1.load_state_dict(torch.load('outputs/9-elem-200-plastic_testfull/models/[6-8x1-3]-9-elem-200-plastic-6-VFs_1.pt'))
model_1.load_state_dict(torch.load('outputs/9-elem-200-plastic_feedback/models/[27-50x1-27]-9-elem-200-plastic-6-VFs_1.pt'))

model_1.eval()
# model_2.eval()
# model_3.eval()
#model = keras.models.load_model('models/ann3', compile=False)

#model.summary()

# Loading data
df_list, _ = load_dataframes(constants.TRAIN_MULTI_DIR)

# Sampling data pass random seed for random sampling
sampled_dfs = data_sampling(df_list, constants.DATA_SAMPLES)

# Merging training data
data = pd.concat(sampled_dfs, axis=0, ignore_index=True)

trials = list(set(data['tag'].values))

for trial in trials:

    trial_df = data[trial]

    # Reorganizing dataset by time increment
    data_by_tag = [df for _, df in trial_df.groupby(['t'])]

    # Concatenating data groups
    trial_df = pd.concat(data_by_tag).reset_index(drop=True)

    X, y, _, _ = select_features_multi(trial_df)
    X_scaled=x_scaler.transform(X)

data_by_t = [[df for _, df in group.groupby(['t'])] for group in data_by_tag]

data_by_batches = list(itertools.chain(*data_by_tag))



mean_squared_error = torchmetrics.R2Score()

print("----------------------\nMSE\n----------------------\n\tSxx\tSyy\tSxy\n")
with torch.no_grad():
    for i, df in enumerate(sampled_dfs):

        X, y, _, _ = select_features_multi(df)
        X_scaled=x_scaler.transform(X)
        # Apply previous validation dataset
        #X_val, y_val, _, _ = standardize_data(X, y, x_scaler, y_scaler)

        #y_pred = model.predict(X_val)

        # y_pred_inv = model.predict(X)
        y_pred_inv = model_1(torch.tensor(X_scaled)).detach().numpy()

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

        sx_pred_var = y_pred_inv[:,0]
        sy_pred_var = y_pred_inv[:,1]
        sxy_pred_var = y_pred_inv[:,2]

        # sx_pred_var = torch.reshape(y_pred_inv_1,(sx_var_abaqus.shape[0],))
        # sy_pred_var = torch.reshape(y_pred_inv_2,(sy_var_abaqus.shape[0],))
        # sxy_pred_var = torch.reshape(y_pred_inv_3,(sxy_var_abaqus.shape[0],)) 

        mse_x = mean_squared_error(sx_pred_var, torch.tensor(sx_var_abaqus))
        mse_y = mean_squared_error(sy_pred_var, torch.tensor(sy_var_abaqus))
        mse_xy = mean_squared_error(sxy_pred_var, torch.tensor(sxy_var_abaqus))

        print("%i\t%0.5f\t%0.5f\t%0.5f" % (i, mse_x, mse_y, mse_xy))

        # sx_pred_var = y_pred[:,0]
        # sy_pred_var = y_pred[:,1]
        # sxy_pred_var = y_pred[:,2]

        # popt, pcov = scipy.optimize.curve_fit(func, exy_var_abaqus, sxy_var_abaqus)
        
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
        
             

       
        #results = model.evaluate(X_val,y_val[:,:3])
        
        #plt.show()
        # elem_list = [1,3,5,7,9]
        # if df['id'][0] in elem_list:
        #     predictions = pd.DataFrame(y_pred_inv, columns=['pred_x','pred_y','pred_xy'])
        #     results = pd.concat([df[['exx_t','eyy_t','exy_t','sxx_t','syy_t','sxy_t']],predictions], axis=1)
        #     results.to_csv('outputs/9-elem-200-plastic_feedback/val/' + df['tag'][0]+'_'+str(df['id'][0])+'.csv', header=True, sep=',',float_format='%.8f')
        #     print('hey')

        
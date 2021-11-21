import numpy as np
import matplotlib.pyplot as plt
import torchmetrics
from constants import *
import torch
from torch import nn
import pandas as pd
import os
from cycler import cycler
from functions import load_dataframes, data_sampling, select_features_multi
from torchmetrics import MeanSquaredError
import imageio
import copy
import io
import cv2

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

    def forward(self, x):

        #x = self.layers[0](x)
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
            
        return self.layers[-1](x)

# Constants
DATA = '9-elem'
DIR_MODEL = 'outputs/' + DATA + '/models/'
DIR_DATA = 'data/training_multi/'+ DATA + '/'
DIR_VAL = 'outputs/' + DATA + '/val/'
ARCH = '[6-8x1-3]'

N_INPUTS = int(ARCH[1])
N_OUTPUTS = int(ARCH[-2])
N_UNITS = int(ARCH.split('-')[1].split('x')[0])
N_LAYERS = int(ARCH.split('-')[1].split('x')[1])


# Update matplotlib params
default_cycler = (cycler(color=["#ef476f","#118ab2","#073b4c"]))

plt.rcParams.update(PARAMS)
plt.rc('axes', prop_cycle=default_cycler)

# Loading datasets
df_list, file_names = load_dataframes(DIR_DATA)

file_names = [file_name.split('/')[-1] for file_name in file_names]

# Sampling data
sampled_dfs = data_sampling(df_list, DATA_SAMPLES)

model_list = []
# Load ANN models
for r, d, f in os.walk(DIR_MODEL):
    for file in sorted(f, key=lambda x: int(x.split('-')[-2])):
        if ARCH in file and '.pt' in file:
            model_list.append(DIR_MODEL + file)

vfs = [int(model.split('-')[-2]) for model in model_list]
vfs_zero = copy.deepcopy(vfs)
vfs_zero.insert(0,0)

pred_dict = dict.fromkeys(vfs_zero)
metric_dict = dict.fromkeys(vfs)
models_dict = dict.fromkeys(vfs)

for i, model in enumerate(model_list):
    ann = NeuralNetwork(N_INPUTS, N_OUTPUTS, N_UNITS, N_LAYERS)
    ann.load_state_dict(torch.load(model))
    ann.eval()
    models_dict[vfs[i]] = ann

# Model metrics
   

elem_ids = [1, 3, 5, 7, 9]

FRAMES = 20

with torch.no_grad():
    
    for k, df in enumerate(sampled_dfs):
        id = list(set(df['id']))[0]
        if id not in elem_ids:
            continue
        print(str(id))
        
        X, y, _, _ = select_features_multi(df)

        sx_abaqus = df['sxx_t']
        sy_abaqus = df['syy_t']
        sxy_abaqus = df['sxy_t']

        ex_abaqus = df['exx_t']
        ey_abaqus = df['eyy_t']
        exy_abaqus = df['exy_t']

        for (key, model) in models_dict.items():

            pred_dict[key] = model(torch.tensor(X.values).float()).detach().numpy()

            sx_pred = pred_dict[key][:,0]
            sy_pred = pred_dict[key][:,1]
            sxy_pred = pred_dict[key][:,2]

            mean_squared_error = torchmetrics.R2Score()

            mse_x = mean_squared_error(torch.tensor(sx_pred), torch.tensor(sx_abaqus)).item()
            mse_y = mean_squared_error(torch.tensor(sy_pred), torch.tensor(sy_abaqus)).item()
            mse_xy = mean_squared_error(torch.tensor(sxy_pred), torch.tensor(sxy_abaqus)).item()

            metric_dict[key] = [mse_x, mse_y, mse_xy]

        
        pred_dict[0] = pred_dict[key] * 0.0

        images = []

        for i in np.arange(0, len(vfs_zero)-1):
        # get current and next y coordinates
            y_x = pred_dict[vfs_zero[i]][:,0]
            y1_x = pred_dict[vfs_zero[i+1]][:,0]

            y_y = pred_dict[vfs_zero[i]][:,1]
            y1_y = pred_dict[vfs_zero[i+1]][:,1]

            y_xy = pred_dict[vfs_zero[i]][:,2]
            y1_xy = pred_dict[vfs_zero[i+1]][:,2]

            # calculate the distance to the next position
            y_path_x = y1_x - y_x

            y_path_y = y1_y - y_y

            y_path_xy = y1_xy - y_xy

            for j in np.arange(0, FRAMES + 1):
                # divide the distance by the number of frames 
                # and multiply it by the current frame number
                y_temp_x = (y_x + (y_path_x / FRAMES) * j)
                y_temp_y = (y_y + (y_path_y / FRAMES) * j)
                y_temp_xy = (y_xy + (y_path_xy / FRAMES) * j)
            
                fig , (ax1, ax2, ax3) = plt.subplots(1,3)
                fig.suptitle(r''+ df['tag'][0].replace('_','\_') + ': element \#' + str(df['id'][0]) + ' - ' + str(vfs_zero[i+1]) + 'VFs', fontsize=14, y=0.95)
                fig.set_size_inches(10, 6)
                fig.subplots_adjust(bottom=0.2, top=0.8)
                
                ax1.plot(ex_abaqus, sx_abaqus, label='ABAQUS', lw=1)
                ax1.set(xlabel=r'$\varepsilon$', ylabel=r'$\sigma_{xx}$ [MPa]')
                ax2.plot(ey_abaqus, sy_abaqus, label='ABAQUS', lw=1)
                ax2.set(xlabel=r'$\varepsilon$', ylabel=r'$\sigma_{yy}$ [MPa]')
                ax3.plot(exy_abaqus, sxy_abaqus, label='ABAQUS', lw=1)
                ax3.set(xlabel=r'$\varepsilon$', ylabel=r'$\tau_{xy}$ [MPa]')

                ax1.plot(ex_abaqus, y_temp_x, label= 'ANN', lw=1)
                ax1.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
                
                #ax1.set_title(r'$\text{R2}=%0.3f$' % (metric_dict[vfs[i]][0]), fontsize=11)
                ax2.plot(ey_abaqus, y_temp_y, label='ANN', lw=1)
                ax2.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
                #ax2.set_title(r'$\text{R2}=%0.3f$' % (metric_dict[vfs[i]][1]), fontsize=11)
                
                ax3.plot(exy_abaqus, y_temp_xy, label='ANN', lw=1)
                ax3.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
                
                #ax3.set_title(r'$\text{R2}=%0.3f$' % (metric_dict[vfs[i]][2]), fontsize=11)
                handles, labels = ax3.get_legend_handles_labels()
                lgd = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5,0.1), ncol=2)


                io_buf = io.BytesIO()
                fig.savefig(io_buf, format='png', dpi=300)
                io_buf.seek(0)
                img_arr = np.frombuffer(io_buf.getvalue(), dtype=np.uint8)
                img = cv2.imdecode(img_arr, 1)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
                
                #fig.canvas.draw()
                # image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
                # images.append(image.reshape(600, 1000, 3)) ## Y×X
                print('Appending image... vf: %i_frame: %i' % (i, j), end='\r', flush=True)
                # build file name and append to list of file names
                #filename = DIR_VAL + f'_{i}_{j}.png'
                #filenames.append(filename)
                # last frame of each viz stays longer
                if (j == FRAMES):
                    for j in range(8):
                        #filenames.append(filename)
                        images.append(img)
                io_buf.close()
                # save img
                #plt.savefig(filename, dpi=300)
                plt.close()

        # Build GIF
        print('Creating gif\n')
        imageio.mimsave(DIR_VAL + '%i.gif' % (k), images)
        # with imageio.get_writer(DIR_VAL + '%i.gif' % (k), mode='I') as writer:
        #     for filename in filenames:
        #         image = imageio.imread(filename)
        #         writer.append_data(image)
        print('Gif saved\n')        
        print('DONE')

 

# for k, df in enumerate(sampled_dfs):

#     if list(set(df['id']))[0] not in elem_ids:
#         continue

#     sx_abaqus = df['sxx_t']
#     sy_abaqus = df['syy_t']
#     sxy_abaqus = df['sxy_t']

#     ex_abaqus = df['exx_t']
#     ey_abaqus = df['eyy_t']
#     exy_abaqus = df['exy_t']

#     images = []

#     for i in np.arange(0, len(vfs_zero)-1):
#         # get current and next y coordinates
#         y_x = pred_dict[vfs_zero[i]][:,0]
#         y1_x = pred_dict[vfs_zero[i+1]][:,0]

#         y_y = pred_dict[vfs_zero[i]][:,1]
#         y1_y = pred_dict[vfs_zero[i+1]][:,1]

#         y_xy = pred_dict[vfs_zero[i]][:,2]
#         y1_xy = pred_dict[vfs_zero[i+1]][:,2]

#         # calculate the distance to the next position
#         y_path_x = y1_x - y_x

#         y_path_y = y1_y - y_y

#         y_path_xy = y1_xy - y_xy

#         for j in np.arange(0, FRAMES + 1):
#             # divide the distance by the number of frames 
#             # and multiply it by the current frame number
#             y_temp_x = (y_x + (y_path_x / FRAMES) * j)
#             y_temp_y = (y_y + (y_path_y / FRAMES) * j)
#             y_temp_xy = (y_xy + (y_path_xy / FRAMES) * j)
        
#             fig , (ax1, ax2, ax3) = plt.subplots(1,3)
#             fig.suptitle(r''+ df['tag'][0].replace('_','\_') + ': element \#' + str(df['id'][0]) + ' - ' + str(vfs_zero[i+1]) + 'VFs', fontsize=14, y=0.95)
#             fig.set_size_inches(10, 6)
#             fig.subplots_adjust(bottom=0.2, top=0.8)
            
#             ax1.plot(ex_abaqus, sx_abaqus, label='ABAQUS', lw=1)
#             ax1.set(xlabel=r'$\varepsilon$', ylabel=r'$\sigma_{xx}$ [MPa]')
#             ax2.plot(ey_abaqus, sy_abaqus, label='ABAQUS', lw=1)
#             ax2.set(xlabel=r'$\varepsilon$', ylabel=r'$\sigma_{yy}$ [MPa]')
#             ax3.plot(exy_abaqus, sxy_abaqus, label='ABAQUS', lw=1)
#             ax3.set(xlabel=r'$\varepsilon$', ylabel=r'$\tau_{xy}$ [MPa]')

            
#             ax1.plot(ex_abaqus, y_temp_x, label= 'ANN', lw=1)
#             ax1.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
            
#             ax1.set_title(r'$\text{MSE}=%0.3f$' % (metric_dict[vfs_zero[i+1]][0]), fontsize=11)
#             ax2.plot(ey_abaqus, y_temp_y, label='ANN', lw=1)
#             ax2.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
#             ax2.set_title(r'$\text{MSE}=%0.3f$' % (metric_dict[vfs_zero[i+1]][1]), fontsize=11)
            
#             ax3.plot(exy_abaqus, y_temp_xy, label='ANN', lw=1)
#             ax3.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
            
#             ax3.set_title(r'$\text{MSE}=%0.3f$' % (metric_dict[vfs_zero[i+1]][2]), fontsize=11)
#             handles, labels = ax3.get_legend_handles_labels()
#             lgd = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5,0.1), ncol=2)


#             io_buf = io.BytesIO()
#             fig.savefig(io_buf, format='png', dpi=300)
#             io_buf.seek(0)
#             img_arr = np.frombuffer(io_buf.getvalue(), dtype=np.uint8)
#             img = cv2.imdecode(img_arr, 1)
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             images.append(img)
            
#             #fig.canvas.draw()
#             # image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
#             # images.append(image.reshape(600, 1000, 3)) ## Y×X
#             print('Appending image... vf: %i_frame: %i' % (i, j), end='\r', flush=True)
#             # build file name and append to list of file names
#             #filename = DIR_VAL + f'_{i}_{j}.png'
#             #filenames.append(filename)
#             # last frame of each viz stays longer
#             if (j == FRAMES):
#                 for j in range(8):
#                     #filenames.append(filename)
#                     images.append(img)
#             io_buf.close()
#             # save img
#             #plt.savefig(filename, dpi=300)
#             plt.close()

#     # Build GIF
#     print('Creating gif\n')
#     imageio.mimsave(DIR_VAL + '%i.gif' % (k), images)
#     # with imageio.get_writer(DIR_VAL + '%i.gif' % (k), mode='I') as writer:
#     #     for filename in filenames:
#     #         image = imageio.imread(filename)
#     #         writer.append_data(image)
#     print('Gif saved\n')
#     print('Removing Images\n')
#     # Remove files
#     for filename in set(filenames):
#         os.remove(filename)
        
#     print('DONE')


print('hey')
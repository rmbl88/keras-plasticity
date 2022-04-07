from readline import parse_and_bind
from sklearn import preprocessing
import pandas as pd
import os
import numpy as np
import copy
import matplotlib.pyplot as plt
import constants
import copy
from constants import FORMAT_PBAR, LOOK_BACK
import torch
from tqdm import tqdm
from torch import nn
from io import StringIO
from torch.nn.utils import (
  parameters_to_vector as Params2Vec,
  vector_to_parameters as Vec2Params
)

# -------------------------------
#        Class definitions
# -------------------------------

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

# EarlyStopping class as in: https://github.com/Bjarten/early-stopping-pytorch/
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

class Element():
    def __init__(self, connect, node_coord, dof) -> None:
        self.id = connect[0]
        self.connect = connect[1:]
        self.node_coord = torch.tensor(node_coord)
        self.global_dof = dof
        self.local_dof = np.arange(len(dof))
    
    def b_el(self, csi=0, eta=0):

        x = [-1,1,1,-1]
        y = [-1,-1,1,1]

        n = list(zip(x,y))

        dN_dcsi = torch.tensor([0.25*(i)*(1+eta*j) for (i,j) in n])
        dN_deta = torch.tensor([0.25*(1+csi*i)*(j) for (i,j) in n])

        dN_csi_eta = torch.stack([dN_dcsi,dN_deta],0)
        J = dN_csi_eta @ self.node_coord

        dN_x_y = torch.linalg.solve(J,dN_csi_eta)

        b_el = torch.zeros(3,len(n)*2)

        x_dof = self.local_dof[::2]
        y_dof = self.local_dof[1::2]

        b_el[0,x_dof] += dN_x_y[0,:]
        b_el[1,y_dof] += dN_x_y[1,:]

        b_el[2,x_dof] += dN_x_y[1,:]
        b_el[2,y_dof] += dN_x_y[0,:]

        return b_el
        

# -------------------------------
#       Method definitions
# ------------------------------

def get_dataset_batches(data_generator, varIndex=0):
    
    return [torch.stack(iter(data_generator).__next__()[varIndex]) for i in range(len(data_generator))]

def param_vector(model):

    params = [param.data for name, param in model.named_parameters() if param.requires_grad and 'weight' in name and 'activation' not in name]

    return params

def param_deltas(model):

    #d_sigma = torch.zeros()
 
    model_orig = copy.deepcopy(model)

    with torch.no_grad():
        
        model_dict = {key: value for key, value in model_orig.state_dict().items() if 'layer' in key and 'weight' in key}
        delta_dict = copy.deepcopy(model_dict)
        
        total_params = sum([value.shape[0]*value.shape[1] for key, value in model_dict.items()])
        
        eval_dicts = [model_orig.state_dict() for param in range(total_params)]

        k = 0
        for key, weight_matrix in model_dict.items():

            param_vector = copy.deepcopy(weight_matrix).flatten()

            for i in range(len(param_vector)):

                param_vector[i] -= 0.1 * param_vector[i]

                delta_dict[key] = param_vector.unflatten(0,weight_matrix.shape)

                eval_dicts[k].update(delta_dict)

    return eval_dicts

def global_strain_disp(elements, n_nodes):
    total_dofs = n_nodes * 2
    b_glob = torch.zeros([3,total_dofs])

    for element in elements:
        b_glob[:,list(element.global_dof)] += element.b_el()
    
    return b_glob

def prescribe_u(b_glob, conditions):
    b = copy.deepcopy(b_glob)
    
    for condition in conditions:
        b[:,condition-1]=0
    
    return b 

def custom_loss(int_work, ext_work):
    
    #return torch.sum(torch.square(y_pred+y_true))
    #return torch.mean(torch.mean(torch.square(y_pred-y_true),1))
    return (1/(int_work.shape[0]*int_work.shape[1]))*torch.sum(torch.sum(torch.abs(torch.sum(torch.sum(int_work,-1,keepdim=True),-2)-ext_work),1))
    #return (1/(4*int_work.shape[0]*int_work.shape[1]))*torch.sum(torch.sum(torch.square(torch.sum(torch.sum(int_work,-1,keepdim=True),-2)-ext_work),1)) 

def global_dof(connect):
    return np.array(sum([[2*i-1,2*i] for i in connect],[]))

def read_mesh(dir):
    
    def get_substring_index(list, sub):
        return next((s for s in list if sub in s), None)

    inp_file = ''
    for r, d, f in os.walk(dir):
        for file in f:
            if '.inp' in file:
                inp_file = dir+file

    lines = []
    with open(inp_file) as f:
        lines = f.readlines()
    f.close()

    start_part = lines.index(get_substring_index(lines,'*Part, name'))
    end_part = lines.index(get_substring_index(lines,'*End Part'))

    lines = lines[start_part:end_part+1]

    start_mesh = lines.index(get_substring_index(lines,'*Node'))
    end_mesh = lines.index(get_substring_index(lines,'*Element, type'))

    start_connect = end_mesh
    end_connect = lines.index(get_substring_index(lines,'*Nset, nset'))

    mesh = ''.join(lines[start_mesh+1:end_mesh]).replace(' ','').split('\n')
    mesh = pd.read_csv(StringIO('\n'.join(mesh)),names=['node','x','y'])

    connect_str = ''.join(lines[start_connect+1:end_connect]).replace(' ','').split('\n')[:-1]

    elem_nodes = len(connect_str[0].split(','))-1

    connectivity = pd.read_csv(StringIO('\n'.join(connect_str)),names=['id']+['n%i'% i for i in range(elem_nodes)])
    dof = [[int(j) for j in i.split(',')][1:] for i in connect_str]
    dof = np.array([sum([[2*i-1,2*i] for i in a],[]) for a in dof])

    return mesh.values, connectivity.values, dof


def plot_history(history, output, is_custom=None, task=None):
    
    if is_custom == None:
        hist = pd.DataFrame(history.history)
        hist['epoch'] = history.epoch
    else:
        hist = history

    plt.rcParams.update(constants.PARAMS)
   
    plt.figure(figsize=(8,6), constrained_layout = True)
    plt.title(task)
    plt.xlabel('Epoch')
    plt.ylabel(r'Mean Square Error [J\textsuperscript{2}]')
    plt.plot(hist['epoch'], hist['loss'], label='Train Error', color='#4b7394')
    plt.plot(hist['epoch'], hist['val_loss'], label = 'Test Error', color='#6db1e2')
    
    plt.legend()
  
    plt.savefig(output + task + '.png', format="png", dpi=600, bbox_inches='tight')

def standardize_data(X, y, f, scaler_x = None, scaler_y = None, scaler_f = None):

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
    
    if scaler_f == None:        
        scaler_f = preprocessing.StandardScaler()
        scaler_f.fit(f)
        f = scaler_f.transform(f)
    else:    
        f = scaler_f.transform(f)

    return X, y, f, scaler_x, scaler_y, scaler_f

def select_features_multi(df):

    #X = df[['exx_t-1dt', 'eyy_t-1dt', 'exy_t-1dt','exx_t', 'eyy_t', 'exy_t']]
    X = df[['exx_t', 'eyy_t', 'exy_t']]
    y = df[['sxx_t','syy_t','sxy_t']]
    f = df[['fxx_t', 'fyy_t', 'fxy_t']]
    coord = df[['dir','id', 'cent_x', 'cent_y','area']]

    return X, y, f, coord


#-----------------------------------
#   DEPRECATED
#-----------------------------------
# def data_sampling(df_list, n_samples, rand_seed=None):

#     sampled_dfs = []

#     for df in df_list:
        
#         if rand_seed != None:
        
#             idx = random.sample(range(0, len(df.index.values)), n_samples)
        
#         else:
        
#             idx = np.round(np.linspace(0, len(df.index.values) - 1, n_samples)).astype(int)
        
#         idx.sort()
#         sampled_dfs.append(df.iloc[idx])

#         sampled_dfs = [df.reset_index(drop=True) for df in sampled_dfs]

#     return sampled_dfs

def drop_features(df, drop_list):

    new_df = df.drop(drop_list, axis=1)

    return new_df

def add_past_step(var_list, lookback, df):

    new_df = copy.deepcopy(df)

    for i in range(lookback):
        for j, vars in enumerate(var_list):
            t_past = df[vars].values[:-(i+1)]
            zeros = np.zeros((i+1,3))
            t_past = np.vstack([zeros, t_past])
            past_vars = [s.replace('_t','_t-'+str(i+1)+'dt') for s in vars]
            t_past = pd.DataFrame(t_past, columns=past_vars)

            new_df = pd.concat([new_df, t_past], axis=1)

    return new_df

def pre_process(df_list):

    var_list = [['sxx_t','syy_t','sxy_t'],['exx_t','eyy_t','exy_t'],['fxx_t','fyy_t','fxy_t']]
    #lookback = 2
    
    new_dfs = []

    # Drop vars in z-direction and add delta_t
    for df in df_list:
        
        new_df = drop_features(df, ['ezz_t', 'szz_t', 'fzz_t'])
        new_dfs.append(new_df)

    if LOOK_BACK > 0:
        # Add past variables
        for i, df in enumerate(tqdm(new_dfs, desc='Loading and processing data',bar_format=FORMAT_PBAR)):

            new_dfs[i] = add_past_step(var_list, LOOK_BACK, df)

    return new_dfs

def load_dataframes(directory):

    file_list = []
    df_list = []

    for r, d, f in os.walk(directory):
        for file in f:
            if '.csv' in file:
                file_list.append(directory + file)

    #headers = ['tag','id','dir','x', 'y', 'area', 't', 'sxx_t', 'syy_t', 'szz_t', 'sxy_t', 'exx_t', 'eyy_t', 'ezz_t', 'exy_t', 'fxx_t', 'fyy_t', 'fzz_t', 'fxy_t']
    #headers = ['tag','id','dir','x', 'y', 't', 'sxx_t', 'syy_t', 'szz_t', 'sxy_t', 'exx_t', 'eyy_t', 'ezz_t', 'exy_t', 'fxx_t', 'fyy_t', 'fzz_t', 'fxy_t']

    # Loading training datasets
    df_list = [pd.read_csv(file, sep=',', index_col=False, header=0) for file in file_list]

    df_list = pre_process(df_list)

    return df_list, file_list
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
import math
import torch.nn.functional as F
from torch.nn.utils import (
  parameters_to_vector as Params2Vec,
  vector_to_parameters as Vec2Params
)

# -------------------------------
#        Class definitions
# -------------------------------
class soft_exponential(nn.Module):
    '''
    Implementation of soft exponential activation.
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    Parameters:
        - alpha - trainable parameter
    References:
        - See related paper:
        https://arxiv.org/pdf/1602.01321.pdf
    Examples:
        >>> a1 = soft_exponential(256)
        >>> x = torch.randn(256)
        >>> x = a1(x)
    '''
    def __init__(self, alpha = None):
        '''
        Initialization.
        INPUT:
            - in_features: shape of the input
            - aplha: trainable parameter
            aplha is initialized with zero value by default
        '''
        super(soft_exponential,self).__init__()
        
        # initialize alpha
        if alpha == None:
            self.alpha = nn.Parameter(torch.tensor(0.0)) # create a tensor out of alpha
        else:
            self.alpha = nn.Parameter(torch.tensor(alpha)) # create a tensor out of alpha
            
        self.alpha.requiresGrad = True # set requiresGrad to true!

    def forward(self, x):
        '''
        Forward pass of the function.
        Applies the function to the input elementwise.
        '''
        if (self.alpha == 0.0):
            return x

        if (self.alpha < 0.0):
            return - torch.log(1 - self.alpha * (x + self.alpha)) / self.alpha

        if (self.alpha > 0.0):
            return (torch.exp(self.alpha * x) - 1)/ self.alpha + self.alpha

class SoftplusLayer(nn.Module):
    r"""Applies a softplus transformation to the incoming data

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(*, H_{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_{in} = \text{in\_features}`.
        - Output: :math:`(*, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(SoftplusLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, F.softplus(self.weight), self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_hidden_layers=1):
        super(NeuralNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size  = hidden_size
        self.n_hidden_layers = n_hidden_layers
        self.output_size = output_size

        self.layers = nn.ModuleList()

        if self.n_hidden_layers == 0:
            self.layers.append(SoftplusLayer(self.input_size,self.output_size,bias=True))
            #self.activation = torch.nn.PReLU()
        else:   
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

        if self.n_hidden_layers == 0:

            return self.layers[0](x)

        else:
            for layer in self.layers[:-1]:
                
                x = self.activation_h(layer(x))
                
            #return self.layers[-1](x)
            return self.activation_o(self.layers[-1](x))

class ICNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_hidden_layers=1):
        super(ICNN, self).__init__()
        self.input_size = input_size
        self.hidden_size  = hidden_size
        self.n_hidden_layers = n_hidden_layers
        self.output_size = output_size

        self.layers = nn.ModuleList()
        self.passthrough = nn.ModuleList()

        for i in range(self.n_hidden_layers):
            if i == 0:
                in_ = self.input_size
                self.layers.append(torch.nn.Linear(in_, self.hidden_size, bias=True))
            else:
                in_ = self.hidden_size
                self.layers.append(SoftplusLayer(in_, self.hidden_size, bias=True))

        self.layers.append(SoftplusLayer(self.hidden_size, self.output_size, bias=True))

        for layer in self.layers[1:]:
            self.passthrough.append(torch.nn.Linear(input_size,layer.in_features,bias=False))

        self.activation = torch.nn.Softplus()

    def forward(self, x):

        xx = self.layers[0](x)
        
        for i,layer in enumerate(self.layers[1:-1]):
            
            xx = self.activation(layer(xx)+self.passthrough[i](x))
            
        #return self.layers[-1](x)
        return self.activation(self.layers[-1](xx))

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
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.5e} --> {val_loss:.5e}).  Saving model ...')
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

        b_el = torch.zeros(3,len(self.local_dof))

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

# def get_dataset_batches(data_generator, varIndex=0):
    
#     return [torch.stack(iter(data_generator).__next__()[varIndex]) for i in range(len(data_generator))]

# def param_vector(model):

#     params = [param.data for name, param in model.named_parameters() if param.requires_grad and 'weight' in name and 'activation' not in name]

#     return params

def param_deltas(model):
    
    model.eval()
    with torch.no_grad():
        
        model_dict = {key: value for key, value in model.state_dict().items()}

        total_params = sum([len(value.flatten()) for key, value in model_dict.items()])
        
        eval_dicts = [model.state_dict() for param in range(total_params)]

        k = 0
        for key, weight_matrix in model_dict.items():
            
            matrix_len = len(weight_matrix.flatten()) 
            
            for i in range(matrix_len):
                
                param_vector = copy.deepcopy(weight_matrix).flatten()
                
                delta_dict = copy.deepcopy(model_dict)

                param_vector[i] -= 0.05 * param_vector[i]

                delta_dict[key] = param_vector.unflatten(0,weight_matrix.shape)

                eval_dicts[k].update(delta_dict)
                
                k += 1

    return eval_dicts

def global_strain_disp(elements, total_dofs, bcs):
    
    g_dof = list(range(total_dofs))
    n_pts = len(elements)
    n_comps = 3
    b_glob = torch.zeros([n_comps * n_pts, total_dofs])

    # Assembly of global strain-displacement matrix
    for i, element in enumerate(elements):
        
        b_glob[n_comps*i:n_comps*i + n_comps, element.global_dof-1] += element.b_el()
    
    b_bar = copy.deepcopy(b_glob)
    bc_fixed = []
    bc_slaves = []
    bc_masters = []

    for edge, props in bcs.items():

        edge_dof_x = list(props['dof'][::2]-1)
        edge_dof_y = list(props['dof'][1::2]-1)

        if edge == 'left' or edge == 'bottom':
        
            master_dof = list(props['dof'][0:2]-1)
            slave_dof = list(props['dof'][2:]-1)

        elif edge == 'right' or edge == 'top':

            master_dof = list(props['dof'][-2:]-1)
            slave_dof = list(props['dof'][:-2]-1)

        # Set bc along x-direction
        if props['cond'][0] == 0:
            pass
        elif props['cond'][0] == 1:
            bc_fixed += edge_dof_x
        elif props['cond'][0] == 2:
            b_bar[:, master_dof[0]] += torch.sum(b_bar[:,slave_dof[::2]],1)
            bc_slaves += slave_dof[::2]
            bc_masters.append(master_dof[0])
        
        # Set bc along y-direction
        if props['cond'][1] == 0:
            pass
        elif props['cond'][1] == 1:
            bc_fixed += edge_dof_y
        elif props['cond'][1] == 2:
            b_bar[:, master_dof[1]] += torch.sum(b_bar[:,slave_dof[1::2]],1)
            bc_slaves += slave_dof[1::2]
            bc_masters.append(master_dof[1])

    # Defining the active degrees of freedom
    actDOFs = list(set(g_dof)-set(sum([bc_fixed,bc_slaves],[])))
    
    # Checking for incompatible boundary conditions
    if len(list(set(bc_masters).intersection(bc_fixed)))!=0:
        raise Exception('Incompatible BCs, adjacent boundary conditions cannot be both fixed/uniform').with_traceback()

    # Discarding redundant boundary conditions
    b_bar = torch.index_select(b_bar, 1, torch.as_tensor(actDOFs))

    # Computing pseudo-inverse strain-displacement matrix
    b_inv = torch.linalg.pinv(b_bar)
    
    return b_glob, b_bar, b_inv, actDOFs

def prescribe_u(u, bcs):
    U = copy.deepcopy(u)
    v_disp = torch.zeros(u.shape[0],u.shape[1],1,2)
    for edge, props in bcs.items():

        edge_dof_x = list(props['dof'][::2]-1)
        edge_dof_y = list(props['dof'][1::2]-1)

        if edge == 'left' or edge == 'bottom':
        
            master_dof = list(props['dof'][0:2]-1)
            slave_dof = list(props['dof'][2:]-1)

        elif edge == 'right' or edge == 'top':

            master_dof = list(props['dof'][-2:]-1)
            slave_dof = list(props['dof'][:-2]-1)

        # Setting bcs along x_direction
        if props['cond'][0] == 0:
            pass
        elif props['cond'][0] == 1:
            U[:,:,edge_dof_x] = 0
        elif props['cond'][0] == 2:
            U[:,:,slave_dof[::2]] = torch.reshape(U[:,:,master_dof[0]],(U.shape[0],U.shape[1],1,1))
        
            v_disp[:,:,:,0] = torch.mean(U[:,:,edge_dof_x],2)

        # Setting bcs along y_direction
        if props['cond'][1] == 0:
            pass
        elif props['cond'][1] == 1:
            U[:,:,edge_dof_y] = 0
        elif props['cond'][1] == 2:
            U[:,:,slave_dof[1::2]] = torch.reshape(U[:,:,master_dof[1]],(U.shape[0],U.shape[1],1,1))
        
            v_disp[:,:,:,1] = torch.mean(U[:,:,edge_dof_y],2)

    return U, v_disp 

def sbvf_loss(int_work, ext_work):
       
    ivw_sort = torch.sort(torch.abs(int_work.detach()).flatten(),descending=True).values
    #ivw_sort = torch.sort(torch.abs(int_work.detach()),1,descending=True).values

    numSteps = math.floor(0.3*len(ivw_sort))
    alpha = torch.mean(ivw_sort[0:numSteps]) * torch.ones((int_work.shape[0],1))
    #alpha = torch.mean(ivw_sort[:,0:numSteps,:],1)

    return torch.sum((1/alpha**2)*torch.sum(torch.square(int_work-ext_work),1))
    #return torch.sum((1/alpha)*torch.sum(torch.abs(int_work-ext_work),1))

def custom_loss(int_work, ext_work):
    
    #return torch.sum(torch.square(y_pred+y_true))
    #return torch.mean(torch.mean(torch.square(y_pred-y_true),1))
    return (1/(int_work.shape[0]*int_work.shape[1]))*torch.sum(torch.sum(torch.abs(torch.sum(torch.sum(int_work,-1,keepdim=True),-2)-ext_work),1))
    #return (1/(4*int_work.shape[0]*int_work.shape[1]))*torch.sum(torch.sum(torch.square(torch.sum(torch.sum(int_work,-1,keepdim=True),-2)-ext_work),1)) 

def global_dof(connect):
    return np.array(sum([[2*i-1,2*i] for i in connect],[])).astype(int)

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
    plt.yscale('log')
    
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
    info = df[['tag','inc']]
    return X, y, f, coord, info

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
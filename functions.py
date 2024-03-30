import csv
import glob
import random
import pandas as pd
import os
import numpy as np
import copy
from constants import FORMAT_PBAR, LOOK_BACK
import torch
from tqdm import tqdm
from torch import nn
import math
import gc
import pyarrow.parquet as pq
from torch.autograd import Variable
import data_transforms
from itertools import chain
from torch.optim.lr_scheduler import _LRScheduler

# -------------------------------
#        Class definitions
# -------------------------------
class weightConstraint(object):
    def __init__(self, cond='plastic'):
        self.cond = cond
        self.count = 0
    def __call__(self, module):

        if hasattr(module,'weight'):

            if (self.cond == 'plastic'):
            
                w=module.weight.data
                w=w.clamp(0.0)
                module.weight.data=w 
                self.count += 1

            else:
                w=module.weight.data
                w=w.clamp(0.0)
                w[:2,-1]=w[:2,-1].clamp(0.0,0.0)
                w[-1,:2]=w[:2,-1].clamp(0.0,0.0)
                module.weight.data=w 

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_hidden_layers=1,b_norm=False):
        super(NeuralNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size  = hidden_size
        self.n_hidden_layers = n_hidden_layers
        self.output_size = output_size

        self.b_norm = b_norm

        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        #self.dropouts = nn.ModuleList()
        
        if self.b_norm:
            self.b_norms = nn.ModuleList()
        #self.parametrizations = nn.ModuleList()
        #self.drop = nn.Dropout(0.05)

        if self.n_hidden_layers == 0:
            self.layers.append(torch.nn.Linear(self.input_size,self.output_size,bias=True))
            #self.activation = torch.nn.PReLU()
        else:   
            for i in range(self.n_hidden_layers):
                if i == 0:
                    in_ = self.input_size
                    out_ = self.hidden_size[i]
                    
                else:
                    in_ = self.hidden_size[i-1]
                    out_= self.hidden_size[i]

                if self.b_norm:    
                    self.b_norms.append(torch.nn.BatchNorm1d(out_,eps=0.1))
                
                self.layers.append(torch.nn.Linear(in_, out_, bias=True))
                
                self.activations.append(torch.nn.ELU())

                #self.dropouts.append(torch.nn.Dropout(0.05))

                
            self.layers.append(torch.nn.Linear(self.hidden_size[-1], self.output_size, bias=True))

    def forward(self, x):

        if self.n_hidden_layers == 0:

            return self.layers[0](x)

        else:
            for i,layer in enumerate(self.layers[:-1]):
                
                if self.b_norm:
                    x = self.activations[i](self.b_norms[i](layer(x)))
                else:
                    x = self.activations[i](layer(x))
                   
            return self.layers[-1](x)

# # LSTMModel class as in https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_lstm_neuralnetwork/
# class LSTMModel(nn.Module):
#     def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
#         super(LSTMModel, self).__init__()
#         # Hidden dimensions
#         self.device = device
#         self.hidden_dim = hidden_dim

#         # Number of hidden layers
#         self.layer_dim = layer_dim

#         # Building LSTM
#         # batch_first=True causes input/output tensors to be of shape
#         # (batch_dim, seq_dim, feature_dim)
#         self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
#         # Readout layer
#         self.fc = nn.Linear(hidden_dim, output_dim)

#     def forward(self, x):
        
#         h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().cuda()
#         # Initialize cell state
#         c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().cuda()
#         # 28 time steps
#         # We need to detach as we are doing truncated backpropagation through time (BPTT)
#         # If we don't, we'll backprop all the way to the start even after going through another batch
#         out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

#         # Index hidden state of last time step
#         # out.size() --> 100, 28, 100
#         # out[:, -1, :] --> 100, 100 --> just want last time step hidden states! 
#         out = self.fc(out[:, -1, :]) 
#         # out.size() --> 100, 10
#         return out

class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.W_query = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_key = nn.Linear(hidden_size, hidden_size, bias=False)
        self.V = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, encoder_outputs, decoder_hidden):
        query = self.W_query(decoder_hidden)
        keys = self.W_key(encoder_outputs)

        energy = torch.tanh(query + keys)
        attention_scores = self.V(energy).squeeze(dim=-1)

        attention_weights = torch.softmax(attention_scores, dim=-1)
        context_vector = torch.sum(attention_weights.unsqueeze(-1) * encoder_outputs, dim=1)

        return context_vector, attention_weights

class GRUModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: list, layer_dim: int, output_dim: int, fc_bias=True, gru_bias=True):
        super(GRUModel, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.gru_bias = gru_bias
        self.fc_bias = fc_bias  
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim if type(hidden_dim) is list else [hidden_dim]

        # GRU layers
        self.gru = nn.GRU(input_dim, self.hidden_dim[0], layer_dim, batch_first=True, bias=self.gru_bias)

        # Attention layer
        #self.attention = AttentionLayer(hidden_dim[0])

        if len(self.hidden_dim)>1:
            
            self.fc_layers = nn.ModuleList()
            # Fully connected layer
            for i in range(len(self.hidden_dim)-1):
                in_ = self.hidden_dim[i]
                out_ = self.hidden_dim[i+1]
                self.fc_layers.append(nn.Linear(in_, out_, bias=self.fc_bias))

            self.fc_layers.append(nn.Linear(self.hidden_dim[-1], output_dim, bias=True))
        else:
            self.fc = nn.Linear(self.hidden_dim[0], output_dim, bias=True)

        self.relu = nn.LeakyReLU()

    # def init_hidden(self, batch_size):
    #     # Initializing hidden state for first input with zeros
    #     if torch.cuda.is_available():
    #         self.h0 = Variable(torch.zeros(self.layer_dim, batch_size, self.hidden_dim[0]).cuda())
    #     else:
    #         self.h0 = Variable(torch.zeros(self.layer_dim, batch_size, self.hidden_dim[0]))

    def forward(self, x):

        if torch.cuda.is_available() and x.is_cuda:
            self.h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim[0], device=torch.device('cuda')))
        else:
            self.h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim[0]))
        
        # Forward propagation by passing in the input and hidden state into the model   
        out, hidden = self.gru(x,self.h0.detach()) # out[batch_size,seq_len,hidden_size]

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        if len(self.hidden_dim) > 1:
            for layer in self.fc_layers[:-1]:
                    
                out = self.relu(layer(out))
                    
            return self.fc_layers[-1](out)
        else:
            return self.fc(self.relu(out))
            

class GRUModelJit(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, fc_bias=True, gru_bias=True):
        super(GRUModelJit, self).__init__()
        # Defining the number of layers and the nodes in each layer
        self.gru_bias = gru_bias
        self.fc_bias = fc_bias  
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim[0]

        # GRU layers
        self.gru = nn.GRU(input_dim, self.hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        #h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).detach()
        x = torch.nn.utils.rnn.pack_sequence(x)
        out, _ = self.gru(x)
        out = out[0][-1,:]
        out = self.relu(out)
        out = self.fc(out)
        
        return out
    
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
        elif score <= self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'\nEarlyStopping counter: {self.counter} out of {self.patience} | Best: {self.val_loss_min:.6e}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'\nValidation loss decreased ({self.val_loss_min:.5e} --> {val_loss:.5e}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

class BaseLoss(nn.modules.Module):

    def __init__(self, device=None, n_losses=1):
        super(BaseLoss, self).__init__()
        self.device = device

        self.train = False

        #self.mse = torch.nn.MSELoss()

        # Record the weights.
        self.n_losses = n_losses
        self.alphas = torch.zeros((self.n_losses,), requires_grad=False).type(torch.FloatTensor).to(self.device)
    
    def to_eval(self):
        self.train = False

    def to_train(self):
        self.train = True
    
    def mse_constraint(self, res):
        return torch.sum(torch.square(res))/(2*res.shape[0])

    def mse(self, pred, targ):
        res = pred-targ
        return torch.sum(torch.square(res))/(2*res.shape[0])
    
    def forward(self, res_tuples):
        
        loss = [self.mse(*res_tuples[i]) if len(res_tuples[i])>1 else self.mse_constraint(*res_tuples[i]) for i in range(len(res_tuples))]
        
        return loss
    
class CoVWeightingLoss(BaseLoss):

    """
        Wrapper of the BaseLoss which weighs the losses to the Cov-Weighting method,
        where the statistics are maintained through Welford's algorithm. But now for 32 losses.
    """

    def __init__(self, mean_decay=None, device=None, n_losses=1):
        super(CoVWeightingLoss, self).__init__(device, n_losses)

        # # How to compute the mean statistics: Full mean or decaying mean.
        # self.mean_decay = True if args.mean_sort == 'decay' else False
        # self.mean_decay_param = args.mean_decay_param
        
        self.mean_decay_param = mean_decay
        self.current_iter = -1
        self.alphas = torch.zeros((self.n_losses,), requires_grad=False).type(torch.FloatTensor).to(self.device)
        self.weighted_losses = []
        self.unweighted_losses = []

        # Initialize all running statistics at 0.
        self.running_mean_L = torch.zeros((self.n_losses,), requires_grad=False).type(torch.FloatTensor).to(self.device)
        self.running_mean_l = torch.zeros((self.n_losses,), requires_grad=False).type(torch.FloatTensor).to(self.device)
        self.running_S_l = torch.zeros((self.n_losses,), requires_grad=False).type(torch.FloatTensor).to(self.device)
        self.running_std_l = None

    def forward(self, res_tuples):
        # Retrieve the unweighted losses.
        unweighted_losses = super(CoVWeightingLoss, self).forward(res_tuples)
        
        # Put the losses in a list. Just for computing the weights.
        L = torch.tensor(unweighted_losses, requires_grad=False).to(self.device)

        #self.l_norm = torch.linalg.vector_norm(L)
        self.unweighted_losses = L

        # If we are doing validation, we would like to return an unweighted loss be able
        # to see if we do not overfit on the training set.
        if not self.train:
            return torch.sum(self.alphas * L)

        # Increase the current iteration parameter.
        self.current_iter += 1
        # If we are at the zero-th iteration, set L0 to L. Else use the running mean.
        L0 = L.clone() if self.current_iter == 0 else self.running_mean_L
        # Compute the loss ratios for the current iteration given the current loss L.
        l = L / L0

        # If we are in the first iteration set alphas to all 1/32
        if self.current_iter <= 1:
            self.alphas = torch.ones((self.n_losses,), requires_grad=False).type(torch.FloatTensor).to(self.device) / self.n_losses
        # Else, apply the loss weighting method.
        else:
            ls = self.running_std_l / self.running_mean_l
            self.alphas = ls / torch.sum(ls)

        # Apply Welford's algorithm to keep running means, variances of L,l. But only do this throughout
        # training the model.
        # 1. Compute the decay parameter the computing the mean.
        if self.current_iter == 0:
            mean_param = 0.0
        elif self.current_iter > 0 and self.mean_decay_param != None:
            mean_param = self.mean_decay_param
        else:
            mean_param = (1. - 1 / (self.current_iter + 1))

        # 2. Update the statistics for l
        x_l = l.clone().detach()
        new_mean_l = mean_param * self.running_mean_l + (1 - mean_param) * x_l
        self.running_S_l += (x_l - self.running_mean_l) * (x_l - new_mean_l)
        self.running_mean_l = new_mean_l

        # The variance is S / (t - 1), but we have current_iter = t - 1
        running_variance_l = self.running_S_l / (self.current_iter + 1)
        self.running_std_l = torch.sqrt(running_variance_l + 1e-8)

        # 3. Update the statistics for L
        x_L = L.clone().detach()
        self.running_mean_L = mean_param * self.running_mean_L + (1 - mean_param) * x_L

        # Get the weighted losses and perform a standard back-pass.
        weighted_losses = [self.alphas[i] * unweighted_losses[i] for i in range(len(unweighted_losses))]
        self.weighted_losses = weighted_losses

        loss = sum(weighted_losses)
        
        return loss
    
class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """
    
    def __init__(self,
                 optimizer : torch.optim.Optimizer,
                 first_cycle_steps : int,
                 cycle_mult : float = 1.,
                 max_lr : float = 0.1,
                 min_lr : float = 0.001,
                 warmup_steps : int = 0,
                 gamma : float = 1.,
                 last_epoch : int = -1
        ):
        assert warmup_steps < first_cycle_steps
        
        self.first_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle_mult = cycle_mult # cycle steps magnification
        self.base_max_lr = max_lr # first max learning rate
        self.max_lr = max_lr # max learning rate in the current cycle
        self.min_lr = min_lr # min learning rate
        self.warmup_steps = warmup_steps # warmup step size
        self.gamma = gamma # decrease rate of max learning rate by cycle
        
        self.cur_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle = 0 # cycle count
        self.step_in_cycle = last_epoch # step size of the current cycle
        
        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)
        
        # set learning rate min_lr
        self.init_lr()
    
    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)
    
    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr)*self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle-self.warmup_steps) \
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch
                
        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
            
# -------------------------------
#       Method definitions
# -------------------------------
def stress_from_cholesky(pred, d_e, t_pts, n_elems, n_tests=1, sbvf_gen=False):

    l = torch.reshape(pred,[n_tests, t_pts, n_elems, pred.shape[-1]])
    L = torch.zeros([n_tests, t_pts, n_elems, 3, 3])

    tril_indices = torch.tril_indices(row=3, col=3, offset=0)
    L[: ,: , :, tril_indices[0], tril_indices[1]] = l[:,:]
    H = L @ torch.transpose(L,3,4)

    d_s = (H @ d_e.reshape([n_tests, t_pts , n_elems, d_e.shape[-1], 1])).squeeze(-1)
    
    if n_tests==1:
        s = torch.cumsum(d_s.squeeze(),0).reshape([-1,3])
        L = L.squeeze()
        H = H.squeeze()
  
    else:
        s = torch.cumsum(d_s,0).reshape([-1,3])

    if sbvf_gen:
        return s
    else:
        return s, L, H

def layer_wise_lr(model, lr_mult=0.99, learning_rate=0.1):
    layer_names = []
    for n,p in model.named_parameters():
        if 'b_norms' not in n:
            layer_names.append(n)

    layer_names.reverse()

    parameters = []
    prev_group_name = '.'.join(layer_names[0].split('.')[:2])

    # store params & learning rates
    for idx, name in enumerate(layer_names):

        # parameter group name
        cur_group_name = '.'.join(name.split('.')[:2])

        # update learning rate
        if cur_group_name != prev_group_name:
            learning_rate *= lr_mult
        prev_group_name = cur_group_name

        # display info
        print(f'{idx}: lr = {learning_rate:.6f}, {name}')

        # append layer parameters
        parameters += [{'params': [p for n, p in model.named_parameters() if n == name and p.requires_grad],
                        'lr':learning_rate}]
    
    return parameters

def drop_features(df, drop_list):

    new_df = df.drop(drop_list, axis=1)

    return new_df

def add_past_step(var_list, lookback, df):

    new_df = copy.deepcopy(df)

    for i in range(lookback):
        for j, vars in enumerate(var_list):
            t = df[vars].values
            t_past = df[vars].values[:-(i+1)]
            zeros = np.zeros((i+1,len(vars)))
            t_past = np.vstack([zeros, t_past])
                
            past_vars = [s.replace('_t','_t'+str(i+1)) for s in vars]
            t_past = pd.DataFrame(t_past, columns=past_vars)

            new_df = pd.concat([new_df, t_past], axis=1)

    return new_df

def preprocess_vars(var_list, df):

    new_df = copy.deepcopy(df)
    
    vars = [
        'ep_1','ep_2',
        'dep_1','dep_2',
        'ep_1_dir','ep_2_dir', 
        'theta_ep',
        's1','s2',
        'ds1','ds2',
        'theta_sp',
    ]

    # Strain, stress and time variables
    e = df[var_list[0]].values
    e[:,-1] *= 0.5

    s = df[var_list[1]].values

    t = np.reshape(df['t'].values,(len(df),1))
    dt = np.diff(t,axis=0)

    # Calculating principal strains and stresses
    eps_princ, ep_angles = get_principal(e)
    s_princ, sp_angles = get_principal(s)
    # Principal stress rate
    #dot_s_princ = np.gradient(s_princ,t.reshape(-1),axis=0)
    dot_s_princ = np.diff(s_princ,axis=0)/dt
    
    # Principal strain rate
    #dot_e_princ = np.gradient(eps_princ,t.reshape(-1),axis=0)
    dot_e_princ = np.diff(eps_princ,axis=0)/dt
    
    # Direction of strain rate
    de_princ_dir = dot_e_princ/(np.reshape(np.linalg.norm(dot_e_princ,axis=1),(dot_e_princ.shape[0],1)))

    de_princ_dir = np.vstack((de_princ_dir,np.array([np.NaN,np.NaN])))
    dot_s_princ = np.vstack((dot_s_princ,np.array([np.NaN,np.NaN])))
    dot_e_princ = np.vstack((dot_e_princ,np.array([np.NaN,np.NaN])))
    
    princ_vars = pd.DataFrame(
        np.concatenate([eps_princ,dot_e_princ,de_princ_dir,ep_angles.reshape(-1,1),s_princ,dot_s_princ,sp_angles.reshape(-1,1)],1),
        columns=vars
    )

    new_df = pd.concat([new_df,princ_vars],axis=1)

    return new_df

def pre_process(df_list):

    var_list = [['exx_t','eyy_t','exy_t'],['sxx_t','syy_t','sxy_t']]
    #var_list = [['sxx_t','syy_t','sxy_t'],['exx_t','eyy_t','exy_t'],['fxx_t','fyy_t','fxy_t']]
    #lookback = 1
    
    new_dfs = []

    # # Drop vars in z-direction and add delta_t
    # for df in df_list:
        
    #     new_df = drop_features(df, ['ezz_t', 'szz_t', 'fzz_t'])
    #     new_dfs.append(new_df)

    if LOOK_BACK > 0:
        # Add past variables
        for i, df in enumerate(tqdm(df_list, desc='Pre-processing data',bar_format=FORMAT_PBAR)):
            #new_dfs[i] = smooth_data(df)
            #new_dfs[i] = add_past_step(var_list, LOOK_BACK, df)
            new_dfs.append(preprocess_vars(var_list, df))
            #new_dfs[i] = to_sequences(df, var_list, LOOK_BACK)

    return new_dfs

def load_dataframes(directory, preproc=True, cols=None):

    file_list = []
    df_list = []

    for r, d, f in os.walk(directory):
        for file in f:
            if ('.csv' or '.parquet' in file) and 'elems' not in file:
                file_list.append(os.path.join(directory, file))
    
    if 'crux' in directory:
        #df_list = [pd.read_parquet(file, columns=use_cols) for file in tqdm(file_list,desc='Reading .csv files',bar_format=FORMAT_PBAR)]
        df_list = [pq.ParquetDataset(file).read_pandas(columns=cols).to_pandas() for file in tqdm(file_list,desc='Importing dataset files',bar_format=FORMAT_PBAR)]
    else:
        df_list = [pd.read_csv(file, sep=',', index_col=False, header=0, engine='c') for file in tqdm(file_list,desc='Reading .csv files',bar_format=FORMAT_PBAR)]

    gc.collect()

    if preproc:
        df_list = pre_process(df_list)    

    return df_list, file_list

def batch_jacobian(y,x):
    
    batch = x.size(0)
    inp_dim = x.size(-1)
    out_dim = y.size(-1)

    grad_output = torch.eye(out_dim).unsqueeze(1).repeat(1,batch,1).requires_grad_(True)
    gradient = torch.autograd.grad(y,x[:,-1],grad_output,retain_graph=True, is_grads_batched=True)
    J = gradient[0][:,:,-1].permute(1,0,2)
    
    return J

def train_test_split(path: str, test_size: float):
    
    with open(path) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader, None)
        trials = list(chain.from_iterable(list(reader)))
    
    trials = random.sample(trials,len(trials))

    test_trials = random.sample(trials, math.ceil(len(trials) * test_size))
    train_trials = list(set(trials).difference(test_trials))
    
    return train_trials, test_trials

def load_data(path, data_cols: list, file_type='parquet'):
    
    file_list = glob.glob(os.path.join(path, f'*.{file_type}'))
    
    df_list = [pq.ParquetDataset(file).read_pandas(columns=data_cols).to_pandas() for file in tqdm(file_list, desc='Importing dataset files',bar_format=FORMAT_PBAR)]

    raw_data = pd.concat(df_list)
    
    return raw_data 

def get_data_stats(path: str, train_trials, data_cols: list, normalize_type: str):

    raw_data = load_data(path, data_cols)

    input_data = raw_data[raw_data['tag'].isin(train_trials)].drop('tag', axis=1).dropna()

    if normalize_type == 'minmax':

        min = torch.min(torch.from_numpy(input_data.values).float(),0).values
        max = torch.max(torch.from_numpy(input_data.values).float(),0).values

        return {'type': normalize_type, 'min': min, 'max': max}
    
    elif normalize_type == 'standard':

        std, mean = torch.std_mean(torch.from_numpy(input_data.values.astype(np.float32)),0)

        return {'type': normalize_type, 'std': std, 'mean': mean}
    
def get_data_transform(normalize_type: str, stat_vars: dict):

    if normalize_type == 'standard':
        
        transform = data_transforms.Normalize(stat_vars['mean'].tolist(), stat_vars['std'].tolist())
    
    elif normalize_type == 'minmax':

        transform = data_transforms.MinMaxScaler(stat_vars['min'], stat_vars['max'])

    return transform

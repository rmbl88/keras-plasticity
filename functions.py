from itertools import permutations, combinations
from numpy.core.fromnumeric import var
from scipy.sparse import coo
from sklearn import preprocessing
import pandas as pd
import os
import numpy as np
import copy
import random
import matplotlib.pyplot as plt
from tensorflow.python.keras.backend import dtype, global_learning_phase_is_set, print_tensor
from tensorflow.python.ops.gen_batch_ops import batch
from tensorflow.python.ops.variables import global_variables
from tensorflow.python.types import internal
import constants
import keras.backend as kb
import tensorflow as tf
import keras
import copy
from tensorflow.python.ops import gen_array_ops
from constants import ELEM_AREA, ELEM_THICK, FORMAT_PBAR, LENGTH
import math
import torch
from tqdm import tqdm

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

def custom_loss(int_work, ext_work):
    
    #return torch.sum(torch.square(y_pred+y_true))
    #return torch.mean(torch.mean(torch.square(y_pred-y_true),1))
    return (1/(int_work.shape[0]*int_work.shape[1]))*torch.sum(torch.sum(torch.abs(torch.sum(torch.sum(int_work,-1,keepdim=True),-2)-ext_work),1))
    #return (1/(4*int_work.shape[0]*int_work.shape[1]))*torch.sum(torch.sum(torch.square(torch.sum(torch.sum(int_work,-1,keepdim=True),-2)-ext_work),1))


def plot_history(history, output, is_custom=None, task=None):
    
    if is_custom == None:
        hist = pd.DataFrame(history.history)
        hist['epoch'] = history.epoch
    else:
        hist = history

    plt.rcParams.update(constants.PARAMS)

    # find position of lowest validation loss
    #minposs = hist['val_loss'].idxmin() + 1
    
    plt.figure(figsize=(8,6), constrained_layout = True)
    plt.title(task)
    plt.xlabel('Epoch')
    plt.ylabel(r'Mean Square Error [J\textsuperscript{2}]')
    plt.plot(hist['epoch'], hist['loss'], label='Train Error', color='#4b7394')
    plt.plot(hist['epoch'], hist['val_loss'], label = 'Test Error', color='#6db1e2')
    #plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')

    plt.legend()
    #plt.show()

    
    # plt.figure()
    # plt.xlabel('Epoch')
    # plt.ylabel(r'Mean Square Error [MPa\textsuperscript{2}]')
    # plt.plot(hist['epoch'], hist['mse'], label='Train Error', color='#4b7394')
    # plt.plot(hist['epoch'], hist['val_mse'], label = 'Test Error', color='#6db1e2')
    # plt.legend()
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
    coord = df[['dir','id', 'x', 'y','area']]

    return X, y, f, coord

def data_sampling(df_list, n_samples, rand_seed=None):

    sampled_dfs = []

    for df in df_list:
        
        if rand_seed != None:
        
            idx = random.sample(range(0, len(df.index.values)), n_samples)
        
        else:
        
            idx = np.round(np.linspace(0, len(df.index.values) - 1, n_samples)).astype(int)
        
        idx.sort()
        sampled_dfs.append(df.iloc[idx])

        sampled_dfs = [df.reset_index(drop=True) for df in sampled_dfs]

    return sampled_dfs

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
    lookback = 2
    
    new_dfs = []

    # Drop vars in z-direction and add delta_t
    for df in df_list:
        
        new_df = drop_features(df, ['ezz_t', 'szz_t', 'fzz_t'])
        new_dfs.append(new_df)

    # Add past variables
    for i, df in enumerate(tqdm(new_dfs, desc='Loading and processing data',bar_format=FORMAT_PBAR)):

        new_dfs[i] = add_past_step(var_list, lookback, df)

    return new_dfs

def load_dataframes(directory):

    file_list = []
    df_list = []

    for r, d, f in os.walk(directory):
        for file in f:
            if '.csv' in file:
                file_list.append(directory + file)

    headers = ['tag','id','dir','x', 'y', 'area', 't', 'sxx_t', 'syy_t', 'szz_t', 'sxy_t', 'exx_t', 'eyy_t', 'ezz_t', 'exy_t', 'fxx_t', 'fyy_t', 'fzz_t', 'fxy_t']
    #headers = ['tag','id','dir','x', 'y', 't', 'sxx_t', 'syy_t', 'szz_t', 'sxy_t', 'exx_t', 'eyy_t', 'ezz_t', 'exy_t', 'fxx_t', 'fyy_t', 'fzz_t', 'fxy_t']

    # Loading training datasets
    df_list = [pd.read_csv(file, names=headers, sep=',', index_col=False) for file in file_list]

    df_list = pre_process(df_list)

    return df_list, file_list


# def plot_learning_curve(train_sizes, train_scores, test_scores):
    
#     train_scores_mean = np.mean(-train_scores, axis=1)
#     train_scores_std = np.std(-train_scores, axis=1)
#     test_scores_mean = np.mean(-test_scores, axis=1)
#     test_scores_std = np.std(-test_scores, axis=1)

#     #plt.rcParams.update(constants.PARAMS)
    
#     plt.xlabel("Training samples")
#     plt.ylabel("Score")    

#     # Plot learning curve
#     plt.grid(alpha=0.1)
#     plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
#                          train_scores_mean + train_scores_std, alpha=0.1,
#                          color="r")
#     plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
#                          test_scores_mean + test_scores_std, alpha=0.1,
#                          color="g")
#     plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
#                  label="Training score")
#     plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
#                  label="Cross-validation score")
#     plt.legend(loc="best")
    
#     # create_folder('prints')    
#     # save_fig(plt, 'prints/', 'learning','curves')
#     plt.show()
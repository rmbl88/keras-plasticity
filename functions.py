from itertools import permutations, combinations
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
from constants import ELEM_AREA, ELEM_THICK, LENGTH
import math
import torch

def custom_loss(y_pred, y_true):
    
    # # Getting batch size
    # batch_size = y_true.shape[0]

    # # Extracting global force and centroid coordinates from dummy labels array
    # f = torch.reshape(y_true[:,3:5][::9,:],[-1,1,2])
    
    # centroids = y_true[:,-2:]
    
    # cent_x = centroids[:,0]
    
    # cent_y = centroids[:,1]

    # # Defining nodal coordinates
    # coords = list(np.arange(0,LENGTH + 1,1))
    # nodes =  torch.tensor(np.vstack(map(np.ravel, np.meshgrid(coords, coords))).T, requires_grad=True)
    
    # n_nodes = nodes.shape[0]
    
    # x = nodes[:,0]
   
    # y = nodes[:,1]

 
    # # Defining the virtual displacement fields
    # virtual_disp = {
    #     1: torch.stack([x/LENGTH, torch.zeros(n_nodes)], axis=1),
    #     2: torch.stack([torch.zeros(n_nodes), y * (torch.square(x) - x * LENGTH) / (LENGTH ** 3)], axis=1),
    #     3: torch.stack([torch.zeros(n_nodes), torch.sin(y*math.pi/LENGTH) * torch.sin(x*math.pi/LENGTH)], axis=1),
    #     4: torch.stack([torch.sin(y*math.pi/LENGTH) * torch.sin(x*math.pi/LENGTH),torch.zeros(n_nodes)], axis=1),
    #     5: torch.stack([x * y * (x - LENGTH) / LENGTH ** 3,torch.zeros(n_nodes)], axis=1),
    #     6: torch.stack([torch.sin(y * math.pi / LENGTH) * (2 * x * LENGTH - 3 * x ** 2) / LENGTH ** 3,torch.zeros(n_nodes)], axis=1)
    # }

    # # Defining virtual strain fields
    # ones = torch.ones(size=(batch_size,), requires_grad=True)
    # zeros = torch.zeros(size=(batch_size,), requires_grad=True)

    # virtual_strain = {
    #     1:torch.stack([ones/LENGTH, zeros, zeros], 1),
    #     2:torch.stack([zeros, (torch.square(cent_x) - cent_x * LENGTH) / LENGTH ** 3, zeros], 1),
    #     3:torch.stack([zeros, (-math.pi/LENGTH) * torch.cos(cent_y*math.pi/LENGTH) * torch.sin(cent_x*math.pi/LENGTH), zeros], axis=1),
    #     4:torch.stack([zeros, (-math.pi/LENGTH) * torch.sin(cent_y*math.pi/LENGTH) * torch.cos(cent_x*math.pi/LENGTH), zeros], axis=1),
    #     5:torch.stack([(2 * cent_x * cent_y - LENGTH) / LENGTH ** 3, zeros, zeros], 1),
    #     6:torch.stack([torch.sin(cent_y * math.pi / LENGTH) * (2 * cent_x * LENGTH - 3 * cent_x ** 2) / LENGTH ** 3, zeros, zeros], 1)
    # }
   
    # # Getting total number of virtual fields
    # total_vfs = list(virtual_disp.keys())

    # # Defining the internal virtual work
    # int_work = dict.fromkeys(total_vfs)

    # for vf, strain in virtual_strain.items():
    #     int_work[vf] = torch.sum(torch.mul(y_pred, strain) * ELEM_AREA, 1, keepdims=True)
    #     int_work[vf] = torch.reshape(torch.sum(torch.reshape(-ELEM_THICK * int_work[vf], [batch_size//9, 9, 1]), axis=1), [batch_size//9, 1])
    
    # # Defining the external virtual work
    # ext_work = dict.fromkeys(total_vfs)

    # for vf, disp in virtual_disp.items():

    #     ext_work[vf] = f * disp[:]
    #     ext_work[vf] = torch.sum(ext_work[vf], 1, keepdims=True)
    #     ext_work[vf] = torch.sum(ext_work[vf], axis=-1)
    
    # loss = torch.mean(torch.tensor([torch.mean(torch.square(torch.sum(int_work[vf] + ext_work[vf], axis=1))) for vf in total_vfs],  requires_grad=True))
    
    return torch.mean(torch.mean(torch.square(y_pred+y_true),1))



# @tf.function
# def custom_loss(y_true, y_pred):
    
#     Getting batch size
#     batch_size = y_true.shape[0]

#     Extracting global force and centroid coordinates from dummy labels array
#     f = tf.reshape(y_true[:,3:5][::9,:],[-1,1,2])
#     centroids = y_true[:,-2:]
#     cent_x = centroids[:,0]
#     cent_y = centroids[:,1]
    
#     Defining nodal coordinates
#     coords = list(np.arange(0,LENGTH + 1,1))
#     nodes =  tf.convert_to_tensor(np.vstack(map(np.ravel, np.meshgrid(coords, coords))).T, dtype='float32')
#     n_nodes = nodes.shape[0]
#     x = nodes[:,0]
#     y = nodes[:,1]
    
#     Defining the virtual displacement fields
#     virtual_disp = {
#         1: tf.stack([x/LENGTH, tf.zeros(n_nodes)], axis=1),
#         2: tf.stack([tf.zeros(n_nodes), y * (tf.square(x) - x * LENGTH) / (LENGTH ** 3)], axis=1),
#         3: tf.stack([tf.zeros(n_nodes), tf.sin(y*math.pi/LENGTH) * tf.sin(x*math.pi/LENGTH)], axis=1),
#         4: tf.stack([tf.sin(y*math.pi/LENGTH) * tf.sin(x*math.pi/LENGTH),tf.zeros(n_nodes)], axis=1),
#         5: tf.stack([x * y * (x - LENGTH) / LENGTH ** 3,tf.zeros(n_nodes)], axis=1),
#         6: tf.stack([tf.sin(y * math.pi / LENGTH) * (2 * x * LENGTH - 3 * x ** 2) / LENGTH ** 3,tf.zeros(n_nodes)], axis=1)
#     }
    
#     Defining virtual strain fields
#     ones = tf.ones(shape=(batch_size,))
#     zeros = tf.zeros(shape=(batch_size,))

#     virtual_strain = {
#         1:tf.stack([ones/LENGTH, zeros, zeros], 1),
#         2:tf.stack([zeros, (tf.square(cent_x) - cent_x * LENGTH) / LENGTH ** 3, zeros], 1),
#         3:tf.stack([zeros, (-math.pi/LENGTH) * tf.cos(cent_y*math.pi/LENGTH) * tf.sin(cent_x*math.pi/LENGTH), zeros], axis=1),
#         4:tf.stack([zeros, (-math.pi/LENGTH) * tf.sin(cent_y*math.pi/LENGTH) * tf.cos(cent_x*math.pi/LENGTH), zeros], axis=1),
#         5:tf.stack([(2 * cent_x * cent_y - LENGTH) / LENGTH ** 3, zeros, zeros], 1),
#         6:tf.stack([tf.sin(cent_y * math.pi / LENGTH) * (2 * cent_x * LENGTH - 3 * cent_x ** 2) / LENGTH ** 3, zeros, zeros], 1)
#     }
    
#     Getting total number of virtual fields
#     total_vfs = list(virtual_disp.keys())

#     Defining the internal virtual work
#     int_work = dict.fromkeys(total_vfs)

#     for vf, strain in virtual_strain.items():
#         int_work[vf] = tf.reduce_sum(tf.multiply(y_pred, strain) * ELEM_AREA, 1, keepdims=True)
#         int_work[vf] = tf.reshape(tf.reduce_sum(tf.reshape(-ELEM_THICK * int_work[vf], [batch_size//9, 9, 1]), axis=1), [batch_size//9, 1])
    
#     Defining the external virtual work
#     ext_work = dict.fromkeys(total_vfs)

#     for vf, disp in virtual_disp.items():

#         ext_work[vf] = f * disp[:]
#         ext_work[vf] = tf.reduce_sum(ext_work[vf], 1, keepdims=True)
#         ext_work[vf] = tf.reduce_sum(ext_work[vf], axis=1)
    
#     loss = tf.reduce_mean([tf.reduce_mean(tf.square(tf.reduce_sum(int_work[vf] + ext_work[vf], axis=1))) for vf in total_vfs])
    
    
#     tf.reduce_sum( tf.multiply( a, b ), 1, keep_dims=True )
#     A = tf.reshape(A, [81//9, 9, 2])
#     A = tf.reduce_sum(A, axis=1)
#     A = tf.reshape(A, [81//9, 2])

#     f = y_true[:,3:5]
#     coord_vector = y_true[:,-2:]

#     x = coord_vector[:,0]
#     y = coord_vector[:,1]
#     zeros = tf.zeros(shape=(x.shape[0],))
#     ones = tf.ones(shape=(x.shape[0],))

#     coords = list(np.arange(0,LENGTH,1))

#     nodes_x = [tf.ones(shape=x.shape[0],) * coord for coord in list(np.arange(0,LENGTH + 1,1))]
#     nodes_y = [tf.ones(shape=x.shape[0],) * coord for coord in list(np.arange(0,LENGTH + 1,1))]

#     nodes = list(zip(nodes_x,nodes_y))

#     u = [[tf.stack([node_x,zeros],1), tf.stack([zeros,node_y],1)] for node_x, node_y in nodes]
#     e = [tf.stack([ones, zeros,zeros], 1), tf.stack([zeros, ones, zeros], 1)]

#     internal_work = [-ELEM_THICK * tf.multiply(y_pred, e_star) * ELEM_AREA for e_star in e]
#     internal_work = [tf.reshape(tf.reduce_sum(tf.reshape(work, [x.shape[0]//9, 9, 3]), axis=1), [x.shape[0]//9, 3]) for work in internal_work]
#     external_work = [[tf.multiply(f, vf) for vf in u_star] for u_star in u]
#     external_work = [tf.reduce_sum([work[i] for work in external_work], axis=0) for i in range(f.shape[1])]
#     external_work = [tf.reshape(tf.reduce_sum(tf.reshape(work, [x.shape[0]//9, 9, 2]), axis=1), [x.shape[0]//9, 2]) for work in external_work]


#     loss = tf.reduce_mean([tf.reduce_mean(tf.square(internal_work[i][:,:-1] + external_work[i])) for i in range(len(internal_work))])
 
#    return loss

def plot_history(history, is_custom=None):
    
    if is_custom == None:
        hist = pd.DataFrame(history.history)
        hist['epoch'] = history.epoch
    else:
        hist = history

    plt.rcParams.update(constants.PARAMS)
    
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel(r'Mean Square Error [MPa\textsuperscript{2}]')
    plt.plot(hist['epoch'], hist['loss'], label='Train Error', color='#4b7394')
    plt.plot(hist['epoch'], hist['val_loss'], label = 'Test Error', color='#6db1e2')
    plt.legend()
    
    # plt.figure()
    # plt.xlabel('Epoch')
    # plt.ylabel(r'Mean Square Error [MPa\textsuperscript{2}]')
    # plt.plot(hist['epoch'], hist['mse'], label='Train Error', color='#4b7394')
    # plt.plot(hist['epoch'], hist['val_mse'], label = 'Test Error', color='#6db1e2')
    # plt.legend()

    plt.show()

def standardize_data(X, y, scaler_x = None, scaler_y = None):

    if scaler_x == None and scaler_y == None:
        scaler_x = preprocessing.MaxAbsScaler()
        scaler_y = preprocessing.MaxAbsScaler()
        #scaler_x = preprocessing.StandardScaler()
        #scaler_y = preprocessing.StandardScaler()

        scaler_x.fit(X)
        scaler_y.fit(y)

    X = scaler_x.transform(X)
    y = scaler_y.transform(y)

    return X, y, scaler_x, scaler_y

def standardize_(var, scaler = None):

    # # Adding dummy column to force dataframe (not needed if there was force along zz)
    # force.insert(2, 'dummy_col', 0.0)

    if scaler == None:
        scaler = preprocessing.MaxAbsScaler()
        scaler.fit(var)

    v = scaler.transform(var)

    return v, scaler

def select_features(df):

    X = df[['fxx_t-dt','fyy_t-dt', 'exx_t-dt','eyy_t-dt','exy_t-dt', 'exx_t', 'eyy_t', 'exy_t']]
    y = df[['sxx_t','syy_t','sxy_t']]

    return X, y

def select_features_multi(df):

    X = df[['exx_t-dt','eyy_t-dt','exy_t-dt', 'exx_t', 'eyy_t', 'exy_t']]
    y = df[['sxx_t','syy_t','sxy_t']]
    f = df[['fxx_t', 'fyy_t', 'fxy_t']]
    coord = df[['id', 'x', 'y']]

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

def add_delta_t(df):

    new_df = copy.deepcopy(df)

    t_initial = df['t'].values[:-1]
    t_final = df['t'].values[1:]

    delta_t = np.append(t_final - t_initial, 0)

    new_df.insert(loc=2, column='dt', value=delta_t)

    return new_df

def add_future_step(var_list, future_var_list, df):

    new_df = copy.deepcopy(df)

    for i, vars in enumerate(var_list):

        if 'fxx' or 'fyy' or 'fxy' in vars:
            continue
        else:
            t_future = df[vars].values[1:]
            t_future = np.vstack([t_future, [0] * len(vars)])

            t_future = pd.DataFrame(t_future, columns=future_var_list[i])

            new_df = pd.concat([new_df, t_future], axis=1)

            new_df.drop(new_df.tail(1).index, inplace = True)

    return new_df

def add_past_step(var_list, past_var_list, df):

    new_df = copy.deepcopy(df)

    for i, vars in enumerate(var_list):

        t_past = df[vars].values[:-1]
        t_past = np.vstack([[0] * len(vars), t_past])

        t_past = pd.DataFrame(t_past, columns=past_var_list[i])

        new_df = pd.concat([new_df, t_past], axis=1)

    return new_df

def pre_process(df_list):

    var_list = [['sxx_t','syy_t','sxy_t'],['exx_t','eyy_t','exy_t'],['fxx_t','fyy_t','fxy_t']]
    #future_var_list = [['sxx_t+dt','syy_t+dt','sxy_t+dt'],['exx_t+dt','eyy_t+dt','exy_t+dt']]
    past_var_list = [['sxx_t-dt','syy_t-dt','sxy_t-dt'],['exx_t-dt','eyy_t-dt','exy_t-dt'],['fxx_t-dt','fyy_t-dt','fxy_t-dt']]

    new_dfs = []

    # Drop vars in z-direction and add delta_t
    for df in df_list:
        
        new_df = drop_features(df, ['ezz_t', 'szz_t', 'fzz_t'])
        #new_dfs.append(add_delta_t(new_df))
        new_dfs.append(new_df)

    # # Add future variables
    # for i, df in enumerate(new_dfs):

    #     new_dfs[i] = add_future_step(var_list, future_var_list, df)

    # Add past variables
    for i, df in enumerate(new_dfs):

        new_dfs[i] = add_past_step(var_list, past_var_list, df)
        print(str(i))

    return new_dfs

def load_dataframes(directory):

    file_list = []
    df_list = []

    for r, d, f in os.walk(directory):
        for file in f:
            if '.csv' in file:
                file_list.append(directory + file)

    headers = ['tag','id', 'x', 'y', 't', 'sxx_t', 'syy_t', 'szz_t', 'sxy_t', 'exx_t', 'eyy_t', 'ezz_t', 'exy_t', 'fxx_t', 'fyy_t', 'fzz_t', 'fxy_t']

    #headers =  ['id', 't', 'sxx_t', 'syy_t', 'szz_t', 'sxy_t', 'exx_t', 'eyy_t', 'ezz_t', 'exy_t', 'fxx_t', 'fyy_t', 'fzz_t', 'fxy_t']

    # Loading training datasets
    df_list = [pd.read_csv(file, names=headers, sep=',') for file in file_list]

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
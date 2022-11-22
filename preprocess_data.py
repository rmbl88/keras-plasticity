from tqdm import tqdm
from functions import load_dataframes
from constants import *

import os
import gc
import copy
import itertools
import torch
import pandas as pd
import numpy as np
import pyarrow.parquet as pq


def load_dataframes(directory):

    file_list = []
    df_list = []

    for r, d, f in os.walk(directory):
        for file in f:
            if '.csv' or '.parquet' in file:
                file_list.append(directory + file)

    # Loading training datasets
    use_cols = ['tag','id','inc','t','area','exx_t','eyy_t','exy_t','sxx_t','syy_t','sxy_t','fxx_t','fyy_t']
    
    if 'crux' in directory:
        df_list = [pq.ParquetDataset(file).read_pandas(columns=use_cols).to_pandas() for file in tqdm(file_list,desc='Importing dataset files',bar_format=FORMAT_PBAR)]
    else:
        df_list = [pd.read_csv(file, sep=',', index_col=False, header=0, engine='c') for file in tqdm(file_list,desc='Reading .csv files',bar_format=FORMAT_PBAR)]

    gc.collect()

    df_list = pre_process(df_list)    

    return df_list, file_list

def pre_process(df_list):

    var_list = [['exx_t','eyy_t','exy_t'],['sxx_t','syy_t','sxy_t']]
        
    new_dfs = []

    if LOOK_BACK > 0:
        # Add past variables
        for i, df in enumerate(tqdm(df_list, desc='Pre-processing data',bar_format=FORMAT_PBAR)):
            #new_dfs[i] = smooth_data(df)
            #new_dfs[i] = add_past_step(var_list, LOOK_BACK, df)
            new_dfs.append(preprocess_vars(var_list, df))
            #new_dfs[i] = to_sequences(df, var_list, LOOK_BACK)

    return new_dfs

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

def get_principal(var, angles=None):  

    if angles is None:
        # Getting principal angles
        angles = 0.5 * np.arctan(2*var[:,-1] / (var[:,0]-var[:,1]))
        angles[np.isnan(angles)] = 0.0

    # Constructing tensors
    tril_indices = np.tril_indices(n=2)
    var_mat = np.zeros((var.shape[0],2,2))

    var[:,[1,2]] = var[:,[2,1]]

    var_mat[:,tril_indices[0],tril_indices[1]] = var[:,:]
    var_mat[:,tril_indices[1],tril_indices[0]] = var[:,:]

    # Rotating tensor to the principal plane
    var_princ_mat = rotate_tensor(var_mat, angles)
    var_princ_mat[abs(var_princ_mat)<=1e-16] = 0.0
    var_princ = var_princ_mat[:,tril_indices[0][:-1],np.array([0,1,0])[:-1]]

    return var_princ, angles

def rotate_tensor(t,theta,is_reverse=False):
    '''Applies a rotation transformation to a given tensor

    Args:
        t (float): The tensor, or batch of tensors, to be transformed
        theta (float): The angle, or angles, of rotation
        is_reverse (bool): Controls if forward or reverse transformation is applied (default is False)

    Returns:
        t_: the rotated tensor
    '''

    r = np.zeros_like(t)
    r[:,0,0] = np.cos(theta)
    r[:,0,1] = np.sin(theta)
    r[:,1,0] = -np.sin(theta)
    r[:,1,1] = np.cos(theta)
    
    if is_reverse:
        t_ = np.transpose(r,(0,2,1)) @ t @ r
    else:
        t_ = r @ t @ np.transpose(r,(0,2,1))
    
    return t_

if __name__ == '__main__':   
    
    # Loading data
    df_list, _ = load_dataframes(os.path.join(TRAIN_MULTI_DIR,'raw/'))

    # Merging training data
    data = pd.concat(df_list, axis=0, ignore_index=True)    
    
    # Reorganizing dataset by tag, subsequent grouping by time increment
    data_by_tag = [df for _, df in data.groupby(['tag'])]
    
    data_by_t = [[df for _, df in group.groupby(['t'])] for group in data_by_tag]
    
    data_by_batches = list(itertools.chain(*data_by_t))
    data_by_batches = [df.sort_values('id') for df in data_by_batches]

    # Getting trial tags     
    trials = [list(set(df['tag']))[0] for df in data_by_tag]
    
    # Getting time points
    time_points = dict.fromkeys(trials)
    
    for i, (k, _) in enumerate(time_points.items()):
        if k == data_by_t[i][0]['tag'].values[0]:
            time_points[k] = len(data_by_t[i])

    # Getting batch sizes
    batch_sizes = dict.fromkeys(trials)

    for i, (k, _) in enumerate(batch_sizes.items()):
        if k == data_by_t[i][0]['tag'].values[0]:
            batch_sizes[k] = len(data_by_t[i][0]) * time_points[k]

    for trial in tqdm(trials, desc='Saving processed data', bar_format=FORMAT_PBAR):
        df = data[data['tag']==trial]
        df.to_parquet(os.path.join(TRAIN_MULTI_DIR,'processed', trial + '.parquet'),compression='brotli')

    trials = pd.DataFrame(trials)
    trials.to_csv(os.path.join(TRAIN_MULTI_DIR,'trials.csv'))
from numpy.core.fromnumeric import var
from numpy.lib.shape_base import column_stack
import pandas as pd
import os
import numpy as np

def data_seletion(df_list, n_samples):

    sampled_dfs = []
    
    for df in df_list:
        idx = np.round(np.linspace(0, len(df.index) - 1, n_samples)).astype(int)
        sampled_dfs.append(df.iloc[idx])
     
    # Merging datasets
    df_train = pd.concat(sampled_dfs, axis=0, ignore_index=True)

    return df_train

def pre_process(df_list):

    var_list = [['sxx_t','syy_t','sxy_t'],['exx_t','eyy_t','exy_t']]
    future_var_list = [['sxx_t+dt','syy_t+dt','sxy_t+dt'],['exx_t+dt','eyy_t+dt','exy_t+dt']]
    past_var_list = [['sxx_t-dt','syy_t-dt','sxy_t-dt'],['exx_t-dt','eyy_t-dt','exy_t-dt']]

    new_dfs_future = []
    new_dfs_past = []

    for df in df_list:
        
        df.drop(['szz_t', 'ezz_t'], axis=1, inplace=True)

        ti = df['t'].values[:-1]
        tf = df['t'].values[1:]

        dt = np.append(tf-ti,0)

        df.insert(loc=2, column='dt', value=dt)

        for i, vars in enumerate(var_list):

            tf = df[vars].values[1:]
            tf = pd.DataFrame(np.vstack([tf,[0,0,0]]), columns=future_var_list[i])

            df = pd.concat([df,tf], axis=1)

        df.drop(df.tail(1).index, inplace = True)

        new_dfs_future.append(df)

    for df in new_dfs_future:

        for j, vars in enumerate(var_list):

            ti = df[vars].values[:-1]
            ti = pd.DataFrame(np.vstack([[0,0,0],ti]), columns=past_var_list[j])

            df = pd.concat([df,ti], axis=1)
        
        new_dfs_past.append(df)

    return new_dfs_past

def load_data(directory):

    file_list = []

    for r, d, f in os.walk(directory):
        for file in f:
            if 'train' and '.csv' in file:
                file_list.append(directory + file)

    headers = ['id', 't', 'sxx_t', 'syy_t', 'szz_t', 'sxy_t', 'exx_t', 'eyy_t', 'ezz_t', 'exy_t']

    # Loading training datasets
    df_list = [pd.read_csv(file, names=headers, sep=',') for file in file_list]

    df_list = pre_process(df_list)

    return df_list
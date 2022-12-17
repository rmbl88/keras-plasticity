import pandas as pd
import numpy as np
import torch
from constants import *
import os
import pyarrow.parquet as pq
from tqdm import tqdm

def pytorch_rolling_window(x, window_size, step_size=1):
    # unfold dimension to make our rolling window
    return x.unfold(0,window_size,step_size)

FEATURES = ['ep_1_dir','ep_2_dir','dep_1','dep_2','ep_1','ep_2']
OUTPUTS = ['ds1','ds2']
INFO = ['tag','inc','t','theta_ep','s1','s2','theta_sp']

file_list=[]

dir = os.path.join(TRAIN_MULTI_DIR,'processed/')
for r, d, f in os.walk(dir):
    for file in f:
        if '.csv' or '.parquet' in file:
            file_list.append(dir + file)

df_list = [pq.ParquetDataset(file).read_pandas(columns=['inc','id','t']+FEATURES+OUTPUTS).to_pandas() for file in tqdm(file_list,desc='Importing dataset files',bar_format=FORMAT_PBAR)]

df = df_list[0].dropna()

t = torch.from_numpy(df['t'].values)
t_pts = len(list(set(t.numpy())))
n_elems = len(set(df['id'].values))

inputs = torch.from_numpy(df[['inc']+FEATURES].values).reshape(t_pts,n_elems,len(FEATURES)+1)



input_width=6
label_width=1
shift=1



w2 = WindowGenerator(train_df=df_list[0],input_width=6, label_width=1, shift=1,label_columns=['ds1','ds2'])

print('hey')
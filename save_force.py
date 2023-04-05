import constants
import joblib
from functions import (GRUModel, read_mesh)
from contour import plot_fields
import matplotlib.pyplot as plt
import pandas as pd
from cycler import cycler
import torch
import numpy as np
import os
from constants import *
import glob
import pyarrow.parquet as pq
from tqdm import tqdm

#-------------------------------------------------------------------------
#                          METHOD DEFINITIONS
#-------------------------------------------------------------------------

def load_pkl(file: str):
    return joblib.load(file)

def scan_ann_files(run: str, dir: str, key: str):

    SCAN_DIR = os.path.join('outputs', dir, 'models')
    
    for f in glob.glob(os.path.join(SCAN_DIR, f'*{run}*')):
    
        if key in f:
            file = f           

    return file

def load_file(run: str, dir: str, key: str):
    
    f = scan_ann_files(run, dir, key)

    return load_pkl(f)

def create_dir(dir: str, root_dir: str):

    ROOT_DIR = root_dir
    DIR = os.path.join(ROOT_DIR, dir)

    try:    
        os.makedirs(DIR)        
    except FileExistsError:
        pass

    return DIR

def get_ann_model(run: str, dir: str):
    
    f = scan_ann_files(run, dir, '.pt')
    
    return torch.load(f)

def load_data(dir: str, ftype: str):

    DIR = os.path.join(dir,'processed')
    
    files = glob.glob(os.path.join(DIR, f'*.{ftype}'))
    
    df_list = [pq.ParquetDataset(file).read_pandas().to_pandas() for file in tqdm(files, desc='Importing dataset files',bar_format=FORMAT_PBAR)]

    return df_list

#--------------------------------------------------------------------------

# Setting Pytorch floating point precision
torch.set_default_dtype(torch.float64)

# Defining ann model to load
RUN = 'whole-puddle-134'

# Defining output directory
DIR = 'crux-plastic_sbvf_abs_direct'

# Creting output directories
F_DIR = create_dir(dir='global_force_ann', root_dir=os.path.join(VAL_DIR_MULTI,'processed'))

RUN_DIR = create_dir(dir=RUN, root_dir=F_DIR)

# Loading model architecture
FEATURES, OUTPUTS, INFO, N_UNITS, H_LAYERS, SEQ_LEN = load_file(RUN, DIR, 'arch.pkl')

# Loading data scaler
MIN, MAX = load_file(RUN, DIR, 'scaler_x.pkl')

MODEL_INFO = {
    'in': FEATURES,
    'out': OUTPUTS,
    'info': INFO,
    'min': MIN,
    'max': MAX
}

# Setting up ANN model
model_1 = GRUModel(input_dim=len(FEATURES),hidden_dim=N_UNITS,layer_dim=H_LAYERS,output_dim=len(OUTPUTS))
model_1.load_state_dict(get_ann_model(RUN, DIR))  
model_1.eval()

# Loading validation data
df_list = load_data(dir=TRAIN_MULTI_DIR, ftype='parquet')

with torch.no_grad():
    
    for i, df in enumerate(pbar := tqdm(df_list, bar_format=FORMAT_PBAR, leave=True)):
        
        # Identifying mechanical test
        tag = df['tag'][0]

        pbar.set_description(f'Saving ANN global force -> {tag}')

        # Number of time steps and number of elements
        n_tps = len(list(set(df['t'])))
        n_elems = len(list(set(df['id'])))
        
        X = df[FEATURES].values
        y = df[OUTPUTS].values
        info = df[INFO]

        pad_zeros = torch.zeros(SEQ_LEN * n_elems, X.shape[-1])
        
        X = torch.cat([pad_zeros, torch.from_numpy(X)], 0)

        x_std = (X - MIN) / (MAX - MIN)
        X_scaled = x_std * (MAX - MIN) + MIN
        
        x = X_scaled.reshape(n_tps + SEQ_LEN,n_elems,-1)
        x = x.unfold(0,SEQ_LEN,1).permute(1,0,3,2)[:,:-1]
        x = x.reshape(-1,*x.shape[2:])
        
        # y = torch.from_numpy(y)
        # y = y.reshape(n_tps,n_elems,-1).permute(1,0,2)
        # y = y.reshape(-1,y.shape[-1])

        # t = torch.from_numpy(info['t'].values).reshape(n_tps, n_elems, 1)

        model_1.init_hidden(x.size(0))
        s = model_1(x) # stress state.

        stress = s.reshape(n_elems, n_tps, -1)
        
        area = torch.from_numpy(df['area'].values).reshape(n_tps,n_elems, -1).permute(1,0,2)
        
        f = torch.sum(stress * area / 30.0, 0).numpy()
        
        f_dataFrame = pd.DataFrame(f,columns=['fxx','fyy','fxy'])
        
        f_dataFrame.to_parquet(os.path.join(RUN_DIR, f'{tag}_force.parquet'),compression='brotli')

        
        
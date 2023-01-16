import constants
import joblib
from functions import (read_mesh, rotate_tensor, NeuralNetwork)
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

def create_run_dir(run: str, dir: str):

    DIR = os.path.join('outputs', dir, 'val')
    
    try:    
        os.makedirs(os.path.join(DIR, run))
        print(f'>> Created folder "{run}" at: "{DIR}"')
    except FileExistsError:
        print(f'>> Folder "{run}" already exists at: "{DIR}"')

def get_ann_model(run: str, dir: str):
    
    f = scan_ann_files(run, dir, '.pt')
    
    return torch.load(f)

def load_data(dir: str, ftype: str):

    DIR = os.path.join(dir,'processed_v')
    
    files = glob.glob(os.path.join(DIR, f'*.{ftype}'))
    
    df_list = [pq.ParquetDataset(file).read_pandas().to_pandas() for file in tqdm(files, desc='Importing dataset files',bar_format=FORMAT_PBAR)]

    return df_list

def get_field_data(df: pd.DataFrame, vars: dict, pred_vars: dict, n_elems: int, n_tps: int):

    T_STEPS = [round((n_tps-1)*0.5), n_tps-1]

    KEYS = sum([[v for k,v in var.items()] for k_,var in vars.items()],[])

    field_dict = {t: {k: {'abaqus': None, 'ann': None, 'err': None} for k in KEYS} for t in T_STEPS}

    for k, d in vars.items():
        for idx, v_name in d.items():
            x = df[v_name].values.reshape(n_tps,n_elems,1)
            y = pred_vars[k][:,idx].reshape(n_tps,n_elems,1)
            for t, d_ in field_dict.items():
                d_[v_name]['abaqus'] = x[t,:]
                d_[v_name]['ann'] = y[t,:]
                d_[v_name]['err'] = np.abs(y[t,:]-x[t,:])

    return field_dict

def import_mesh(dir: str):

    mesh, connectivity, _ = read_mesh(dir)

    nodes = mesh[:,1:]
    connectivity = connectivity[:,1:] - 1

    return nodes, connectivity


def get_re(pred,real):

    '''Calculates the Relative error between a prediction and a real value.'''
    
    re = np.abs(pred-real)*100/np.abs(real)

    return re

#--------------------------------------------------------------------------

# Initializing Matplotlib settings
plt.rcParams.update(constants.PARAMS)
default_cycler = (cycler(color=["#ef476f","#118ab2","#073b4c"]))
plt.rc('axes', prop_cycle=default_cycler)

# Setting Pytorch floating point precision
torch.set_default_dtype(torch.float64)

# Defining ann model to load
RUN = 'logical-fire-305'

# Defining output directory
DIR = 'crux-plastic_sbvf_abs_direct'

# Importing mesh
NODES, CONNECT = import_mesh(TRAIN_MULTI_DIR)

# Loading model architecture
FEATURES, OUTPUTS, INFO, N_UNITS = load_file(RUN, DIR, 'arch.pkl')

# Loading data scaler
STD, MEAN = load_file(RUN, DIR, 'scaler_x.pkl')

ELEMS_VAL = pd.read_csv(os.path.join(VAL_DIR_MULTI,'elems_val.csv'), header=None)[0].to_list()

MODEL_INFO = {
    'in': FEATURES,
    'out': OUTPUTS,
    'info': INFO,
    'std': STD,
    'mean': MEAN
}

# Setting up ANN model
model_1 = NeuralNetwork(len(FEATURES), len(OUTPUTS), N_UNITS, len(N_UNITS))
model_1.load_state_dict(get_ann_model(RUN, DIR))    
model_1.eval()

# Loading validation data
df_list = load_data(dir=VAL_DIR_MULTI, ftype='parquet')

print("--------------------------------------\n\tMean Absolute Error\n--------------------------------------")
with torch.no_grad():
    last_tag = ''
    for i, df in enumerate(df_list):
        
        tag = df['tag'][0]
        
        n_tps = len(list(set(df['t'])))
        n_elems = len(list(set(df['id'])))

        if tag != last_tag:
            print("\n%s\t\tSxx\tSyy\tSxy\tS1\tS2\n" % (tag))
            last_tag = tag
        
        X = df[FEATURES].iloc[:-n_elems].fillna(0.0).values
        y = df[OUTPUTS].iloc[:-n_elems].values
        info = df[INFO]

        X_scaled = (torch.from_numpy(X) - MEAN) / STD
        y = torch.from_numpy(y)

        theta_ep = torch.from_numpy(info[['theta_ep']].values)
        theta_sp = torch.from_numpy(info[['theta_sp']].values)

        t = torch.from_numpy(info['t'].values).reshape(n_tps, n_elems, 1)
        dt = torch.diff(t,dim=0)
 
        s_rate = model_1(X_scaled) # stress rate.
        
        s_princ = torch.zeros(n_tps,n_elems, s_rate.shape[-1])
        
        s_princ[1:,:,:] = s_rate.reshape(n_tps-1,n_elems,s_rate.shape[-1])*dt
        s_princ = torch.cumsum(s_princ,0).reshape(-1, s_rate.shape[-1])

        s_princ_mat = torch.zeros([s_princ.shape[0],2,2])        
        s_princ_mat[:,0,0] = s_princ[:,0]
        s_princ_mat[:,1,1] = s_princ[:,1]

        s = rotate_tensor(s_princ_mat.numpy(),theta_sp.reshape(-1).numpy(),is_reverse=True)
        s = torch.from_numpy(s[:,[0,1,1],[0,1,0]])

        #------------------------------------------------------------------------------------
        #                       PREPARING FIELD DATA FOR CONTOUR PLOT
        #------------------------------------------------------------------------------------
        vars = {'s': {0: 'sxx_t', 1:'syy_t', 2:'sxy_t'}, 's_p': {0:'s1', 1:'s2'}}
        
        pred_vars = {'s': s.numpy(), 's_p': s_princ.numpy(), 'ds': s_rate.numpy()}

        fields = get_field_data(df, vars, pred_vars, n_elems, n_tps)

        plot_fields(NODES, CONNECT, fields)

        #------------------------------------------------------------------------------------

        # Strain values - Abaqus
        ex_abaqus = df['exx_t'].values.reshape(-1,1)
        ey_abaqus = df['eyy_t'].values.reshape(-1,1)
        exy_abaqus = 0.5*df['exy_t'].values.reshape(-1,1)

        # Stress values - Abaqus
        sx_abaqus = df['sxx_t'].values.reshape(-1,1)
        sy_abaqus = df['syy_t'].values.reshape(-1,1)
        sxy_abaqus = df['sxy_t'].values.reshape(-1,1)

        # Stress predictions - ANN
        sx_pred = s[:,0].reshape(-1,1)
        sy_pred = s[:,1].reshape(-1,1)
        sxy_pred= s[:,2].reshape(-1,1)

        ###################################################################################
        #                             MEAN ABSOLUTE ERRORS
        ###################################################################################

        mre_sx = get_re(sx_pred, sx_abaqus).reshape(-1,1)
        mre_sy = get_re(sy_pred, sy_abaqus).reshape(-1,1)
        mre_sxy = get_re(sxy_pred, sxy_abaqus).reshape(-1,1)

        ###################################################################################

        # Principal strain - Abaqus
        e1_abaqus = X['ep_1'].values.reshape(-1,1)
        e2_abaqus = X['ep_2'].values.reshape(-1,1)
        
        # Principal stress - Abaqus
        y_princ = torch.from_numpy(df[['s1','s2']].values)
        y1_abaqus = y_princ[:,0].numpy().reshape(-1,1)
        y2_abaqus = y_princ[:,1].numpy().reshape(-1,1)

        # Principal stress predictions - ANN
        s1_pred = s_princ[:,0].numpy().reshape(-1,1)
        s2_pred = s_princ[:,1].numpy().reshape(-1,1)

        ###################################################################################
        #                             MEAN RELATIVE ERRORS
        ###################################################################################

        mre_s1 = get_re(s1_pred, y1_abaqus).reshape(-1,1)
        mre_s2 = get_re(s2_pred, y2_abaqus).reshape(-1,1)

        ###################################################################################

        # Principal strain rate - Abaqus
        de1_abaqus = X['dep_1'].values.reshape(-1,1)
        de2_abaqus = X['dep_2'].values.reshape(-1,1)

        # Principal stress rate - Abaqus
        # dy_princ = np.gradient(y_princ,t.reshape(-1),axis=0)
        dy_1 = y[:,0].reshape(-1,1).numpy()
        dy_1 = np.vstack((dy_1,np.array([np.NaN])))
        dy_2 = y[:,1].reshape(-1,1).numpy()
        dy_2 = np.vstack((dy_2,np.array([np.NaN])))

        # Principal stress rate predictions
        ds1_pred = s_rate[:,0].numpy().reshape(-1,1)
        ds1_pred = np.vstack((ds1_pred,np.array([np.NaN])))
        ds2_pred = s_rate[:,1].numpy().reshape(-1,1)
        ds2_pred = np.vstack((ds2_pred,np.array([np.NaN])))

        print("Elem #%i\t\t\t%0.3f\t%0.3f\t%0.3f\t%0.3f\t%0.3f" % (df['id'].iloc[0], np.mean(mre_sx), np.mean(mre_sy), np.mean(mre_sxy), np.mean(mre_s1), np.mean(mre_s2)))

        cols = ['e_xx','e_yy','e_xy','s_xx','s_yy','s_xy','s_xx_pred','s_yy_pred','s_xy_pred','e_1','e_2','s_1','s_2','s_1_pred','s_2_pred','de_1','de_2','ds_1','ds_2','dy_1','dy_2','mre_sx','mre_sy','mre_sxy','mre_s1','mre_s2']

        if df['id'].iloc[0] in ELEMS_VAL:
            
            res = np.concatenate([ex_abaqus,ey_abaqus,exy_abaqus,sx_abaqus,sy_abaqus,sxy_abaqus,sx_pred,sy_pred,sxy_pred,e1_abaqus,e2_abaqus,y1_abaqus,y2_abaqus,s1_pred,s2_pred,de1_abaqus,de2_abaqus,ds1_pred,ds2_pred,dy_1,dy_2,mre_sx, mre_sy, mre_sxy, mre_s1, mre_s2], axis=1)
            
            results = pd.DataFrame(res, columns=cols)
            
            results.to_csv(f'outputs/{DIR}/val/{RUN}/' + df['tag'].iloc[0]+'_'+str(df['id'].iloc[0])+'.csv', header=True, sep=',', float_format='%.6f')
            
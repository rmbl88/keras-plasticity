import matplotlib
from matplotlib.offsetbox import AnchoredText
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
from torch.autograd.functional import jacobian

from vfm import get_ud_vfs

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

def get_field_data(abaqus, vars: dict, pred_vars: dict, n_elems: int, n_tps: int):

    T_STEPS = [n_tps-1]

    KEYS = sum([[v for k,v in var.items()] for k_,var in vars.items()],[])

    field_dict = {t: {k: {'abaqus': None, 'ann': None, 'err': None} for k in KEYS} for t in T_STEPS}

    for k, d in vars.items():
        for idx, v_name in d.items():
            if v_name == 'ivw':
                x = np.sum(abaqus[k],-1, keepdims=True)
                y = np.sum(pred_vars[k],-1, keepdims=True)
            else:
                x = abaqus[k][:,:,idx]
                y = pred_vars[k][:,:,idx]

            for t, d_ in field_dict.items():
                d_[v_name]['abaqus'] = x[:,t]
                d_[v_name]['ann'] = y[:,t]
                d_[v_name]['err'] = np.abs(x[:,t]-y[:,t])

    return field_dict

def import_mesh(dir: str):

    mesh, connectivity, _ = read_mesh(dir)

    nodes = mesh[:,1:]
    connectivity = connectivity[:,1:] - 1

    return nodes, connectivity

def get_re(pred,real):

    '''Calculates the Relative error between a prediction and a real value.'''
    
    re = np.abs(pred-real)*100/(1+np.abs(real))

    return re

def get_mises(s_x, s_y, s_xy):
    return np.sqrt(np.square(s_x)+np.square(s_y)-s_x*s_y+3*np.square(s_xy))

def plot_vw(vars, out_dir, vf=None):

    def set_anchored_text(mean_e,median_e,max_e,min_e,frameon=True,loc='lower right'):
        at = AnchoredText(f'Mean: {np.round(mean_e,3)}\nMedian: {np.round(median_e,3)}\nMax.: {np.round(max_e,3)}\nMin.: {np.round(min_e,3)}', loc=loc, frameon=frameon,prop=dict(fontsize=6.5))
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        at.patch.set_linewidth(0.55)
        return at

    matplotlib.use('Agg')
    plt.rcParams.update(PARAMS)

    n_subplots = len(vars.keys())*2
    fig, axs = plt.subplots(2,n_subplots//2)
    fig.set_size_inches(9.6,5.4)
    fig.subplots_adjust(wspace=0.275)

    for i, (k, v) in enumerate(vars.items()):
        axs[0,i].plot(v['ivw'][0], label=v['ivw'][1])
        axs[0,i].plot(v['evw'][0], label=v['evw'][1])
        axs[0,i].legend(loc='best')
        
        err = torch.abs(v['ivw'][0]-v['evw'][0])
        axs[1,i].plot(err,color='k')
        axs[1,i].add_artist(set_anchored_text(torch.mean(err),torch.median(err),torch.max(err),torch.min(err)))

    plt.savefig(os.path.join(out_dir, f'v_work_VF-{vf}.pdf'), format="pdf", dpi=600, bbox_inches='tight')
    
    #plt.show()
    #print('hey')

def plot_vw_elem(vars, out_dir, vf=None):

    def set_anchored_text(mean_e,median_e,max_e,min_e,frameon=True,loc='lower right'):
        at = AnchoredText(f'Mean: {np.round(mean_e,3)}\nMedian: {np.round(median_e,3)}\nMax.: {np.round(max_e,3)}\nMin.: {np.round(min_e,3)}', loc=loc, frameon=frameon,prop=dict(fontsize=6.5))
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        at.patch.set_linewidth(0.55)
        return at

    matplotlib.use('Agg')
    plt.rcParams.update(PARAMS)

    n_subplots = 3
    fig, axs = plt.subplots(2,3)
    fig.set_size_inches(9.6,5.4)
    fig.subplots_adjust(wspace=0.275)

    for i, (k, v) in enumerate(vars.items()):
        for j in range(3):
            axs[0,j].plot(v['ann'][0][:,j], label=v['ann'][1])
            axs[0,j].plot(v['abaqus'][0][:,j], label=v['abaqus'][1])
            axs[0,j].legend(loc='best')
        
            err = torch.abs(v['ann'][0][:,j]-v['abaqus'][0][:,j])
            axs[1,j].plot(err,color='k')
            axs[1,j].add_artist(set_anchored_text(torch.mean(err),torch.median(err),torch.max(err),torch.min(err)))
    elem = v['ann'][1].split('_')[-1]
    plt.savefig(os.path.join(out_dir, f'v_work_VF-{vf}_elem-{elem}.pdf'), format="pdf", dpi=600, bbox_inches='tight')       

#--------------------------------------------------------------------------

# Initializing Matplotlib settings
# plt.rcParams.update(constants.PARAMS)
# default_cycler = (cycler(color=["#ef476f","#118ab2","#073b4c"]))
# plt.rc('axes', prop_cycle=default_cycler)

# Setting Pytorch floating point precision
torch.set_default_dtype(torch.float64)

# Defining ann model to load
#RUN = 'solar-planet-147'
RUN = 'distinctive-shape-42'
#RUN = 'whole-puddle-134'
#RUN = 'rural-rain-45'

# Defining output directory
DIR = 'crux-plastic_sbvf_abs_direct'
#DIR = 'sbvfm_indirect_crux_gru'

# Creting output directories
RUN_DIR = create_dir(dir=RUN, root_dir=os.path.join('outputs', DIR, 'val'))

# Importing mesh
NODES, CONNECT = import_mesh(TRAIN_MULTI_DIR)

CENTROIDS = pd.read_csv(os.path.join(TRAIN_MULTI_DIR,'centroids.csv'), usecols=['cent_x','cent_y']).values

SURF_COORDS = [max(NODES[:,0]),max(NODES[:,1])]

WIDTH = max(NODES[:,0]) - min(NODES[:,0])
HEIGHT = max(NODES[:,1]) - min(NODES[:,1])

TOTAL_VFS, V_DISP, V_STRAIN = get_ud_vfs(CENTROIDS, SURF_COORDS, WIDTH, HEIGHT)

V_DISP = torch.from_numpy(V_DISP)
V_STRAIN = torch.from_numpy(V_STRAIN)

# Loading model architecture
FEATURES, OUTPUTS, INFO, N_UNITS, H_LAYERS, SEQ_LEN = load_file(RUN, DIR, 'arch.pkl')

# Loading data scaler
#MIN, MAX = load_file(RUN, DIR, 'scaler_x.pkl')
SCALER_DICT = load_file(RUN, DIR, 'scaler_x.pkl')

# MODEL_INFO = {
#     'in': FEATURES,
#     'out': OUTPUTS,
#     'info': INFO,
#     'min': MIN,
#     'max': MAX
# }

DRAW_CONTOURS = False
TAG = 'vfm_test_crux'

# Setting up ANN model
model_1 = GRUModel(input_dim=len(FEATURES),hidden_dim=N_UNITS,layer_dim=H_LAYERS,output_dim=len(OUTPUTS))
model_1.load_state_dict(get_ann_model(RUN, DIR))    
model_1.eval()

# Loading validation data
df_list = load_data(dir=TRAIN_MULTI_DIR, ftype='parquet')

cols = ['e_xx','e_yy','e_xy','s_xx','s_yy','s_xy','s_xx_pred','s_yy_pred','s_xy_pred',
        'mre_sx','mre_sy','mre_sxy']

with torch.no_grad():
    last_tag = ''
    for i, df in enumerate(df_list):
        
        # Identifying mechanical test
        tag = df['tag'][0]

        #f_ann = pq.ParquetDataset(os.path.join(TRAIN_MULTI_DIR,'processed','global_force_ann', RUN, tag + '_force.parquet')).read_pandas(columns=['fxx','fyy']).to_pandas()

        #f_ann = torch.from_numpy(f_ann.values)

        f = torch.from_numpy(df[['fxx_t','fyy_t']].values)
        #------------------------------------------------------------------------------------
        #                       CREATING OUTPUT DIRECTORIES
        #------------------------------------------------------------------------------------
        TRIAL_DIR = create_dir(dir=tag+'_vfm',root_dir=RUN_DIR)
        
        DATA_DIR = create_dir(dir='data_vfm',root_dir=TRIAL_DIR)

        PLOT_DIR = create_dir(dir='plots_vfm',root_dir=TRIAL_DIR)

        if DRAW_CONTOURS:
            COUNTOUR_DIR = create_dir(dir='contour_vfm', root_dir=TRIAL_DIR)
        #------------------------------------------------------------------------------------
        
        # Number of time steps and number of elements
        n_tps = len(list(set(df['t'])))
        n_elems = len(list(set(df['id'])))
        
        X = df[FEATURES].values
        y = df[OUTPUTS].values
        info = df[INFO]

        #pad_zeros = torch.zeros(SEQ_LEN * n_elems, X.shape[-1])
        pad_zeros = torch.zeros((SEQ_LEN-1) * n_elems, X.shape[-1])
        
        X = torch.cat([pad_zeros, torch.from_numpy(X)], 0)
        
        if SCALER_DICT['type'] == 'standard':
            X_scaled = (X-SCALER_DICT['stat_vars'][1])/SCALER_DICT['stat_vars'][0]

        # x_std = (X - MIN) / (MAX - MIN)
        # X_scaled = x_std * (MAX - MIN) + MIN

        #x = X_scaled.reshape(n_tps + SEQ_LEN,n_elems,-1)
        #x = x.unfold(0,SEQ_LEN,1).permute(1,0,3,2)[:,:-1]
        x = X_scaled.reshape(n_tps + SEQ_LEN-1, n_elems, -1)
        x = x.unfold(0,SEQ_LEN,1).permute(1,0,3,2)
        x = x.reshape(-1,*x.shape[2:])
        
        y = torch.from_numpy(y)
        #y = y.reshape(n_tps,n_elems,-1)[SEQ_LEN-1:].permute(1,0,2)
        y = y.reshape(n_tps,n_elems,-1).permute(1,0,2)
        y = y.reshape(-1,y.shape[-1])

        # theta_ep = torch.from_numpy(info[['theta_ep']].values)
        # theta_sp = torch.from_numpy(info[['theta_sp']].values)

        t = torch.from_numpy(info['t'].values).reshape(n_tps, n_elems, 1)
        #dt = torch.diff(t,dim=0)

        s = model_1(x) # stress rate.
        
        # s_princ = torch.zeros(n_tps,n_elems, s_rate.shape[-1])
        
        # s_princ[1:,:,:] = s_rate.reshape(n_tps-1,n_elems,s_rate.shape[-1])*dt
        # s_princ = torch.cumsum(s_princ,0).reshape(-1, s_rate.shape[-1])

        # s_princ_mat = torch.zeros([s_princ.shape[0],2,2])        
        # s_princ_mat[:,0,0] = s_princ[:,0]
        # s_princ_mat[:,1,1] = s_princ[:,1]

        # s = rotate_tensor(s_princ_mat.numpy(),theta_sp.reshape(-1).numpy(),is_reverse=True)
        # s = torch.from_numpy(s[:,[0,1,1],[0,1,0]])

        #------------------------------------------------------------------------------------
        #                              RESHAPING DATA
        #------------------------------------------------------------------------------------

        s = s.reshape(n_elems,n_tps,-1)
        y = y.reshape(n_elems,n_tps,-1)
        f = f.reshape(n_tps,n_elems,2).permute(1,0,2)[0,:]

        a = torch.from_numpy(df[df['t']==0]['area'].values).unsqueeze(-1)

        w_int_ann = torch.sum(torch.sum(s*V_STRAIN.unsqueeze(2)*a.unsqueeze(1),-1,keepdim=True),1)

        w_int_ann_ = s*V_STRAIN.unsqueeze(2)*a.unsqueeze(1)

        w_int = torch.sum(torch.sum(y*V_STRAIN.unsqueeze(2)*a.unsqueeze(1),-1,keepdim=True),1)

        w_int_ = y*V_STRAIN.unsqueeze(2)*a.unsqueeze(1)

        f_ann = torch.sum(s*a.unsqueeze(1)/30,0)[:,:2]
        w_ext_ann = torch.sum(f_ann*V_DISP.unsqueeze(1),-1,keepdim=True)

        w_ext = torch.sum(f*V_DISP.unsqueeze(1),-1,keepdim=True)
        
        #------------------------------------------------------------------------------------
        #                       PREPARING FIELD DATA FOR CONTOUR PLOT
        #------------------------------------------------------------------------------------
        if DRAW_CONTOURS and tag == TAG:

            vars = {'ivw': {0: 'ivw_xx', 1:'ivw_yy', 2:'ivw_xy', 3: 'ivw'}}
            
            for vf in range(w_int_ann_.shape[0]):
                pred_vars = {'ivw': w_int_ann_[vf].numpy()}
                abaqus = {'ivw': w_int_[vf].numpy()}

                fields = get_field_data(abaqus, vars, pred_vars, n_elems, n_tps)
            
                plot_fields(NODES, CONNECT, fields, out_dir=COUNTOUR_DIR, tag=tag, vf=vf)

        vars_dict = {vf: {
            'ann':{
                'ivw':(w_int_ann[vf],f'w_int_ann_vf_{vf}'),
                'evw':(w_ext_ann[vf], f'w_ext_ann_vf_{vf}')
                }, 
            'abaqus':{
                'ivw':(w_int[vf], f'w_int_abaqus_vf_{vf}'),
                'evw':(w_ext[vf], f'w_ext_abaqus_vf_{vf}')}} for vf in range(w_ext.shape[0])
        }

        for vf, vars in vars_dict.items():
            plot_vw(vars, out_dir=PLOT_DIR, vf=vf)

        elem=10
        vars_dict = {vf: {
            'ivw':{
                'ann':(w_int_ann_[vf][elem-1],f'w_int_ann_vf_{vf}_elem-{elem}'),
                'abaqus':(w_int_[vf][elem-1], f'w_int_abaqus_vf_{vf}_elem-{elem}')
                }
            } for vf in range(w_ext.shape[0])
        }

        for vf, vars in vars_dict.items():
            plot_vw_elem(vars, out_dir=PLOT_DIR, vf=vf)
        
    print('hey')

        # for elem in []:
        #     idx = elem - 1
        #     # Strain values - Abaqus
        #     ex_abaqus = df[df['id']==elem]['exx_t'].values.reshape(-1,1)
        #     ey_abaqus = df[df['id']==elem]['eyy_t'].values.reshape(-1,1)
        #     exy_abaqus = 0.5*df[df['id']==elem]['exy_t'].values.reshape(-1,1)

        #     # Stress values - Abaqus
        #     sx_abaqus = df[df['id']==elem]['sxx_t'].values.reshape(-1,1)
        #     sy_abaqus = df[df['id']==elem]['syy_t'].values.reshape(-1,1)
        #     sxy_abaqus = df[df['id']==elem]['sxy_t'].values.reshape(-1,1)

        #     # Stress predictions - ANN
        #     sx_pred = s[idx,:,0].reshape(-1,1)
        #     sy_pred = s[idx,:,1].reshape(-1,1)
        #     sxy_pred= s[idx,:,2].reshape(-1,1)

        #     ###################################################################################
        #     #                             RELATIVE ERRORS
        #     ###################################################################################

        #     mre_sx = get_re(sx_pred, sx_abaqus).numpy().reshape(-1,1)
        #     mre_sy = get_re(sy_pred, sy_abaqus).numpy().reshape(-1,1)
        #     mre_sxy = get_re(sxy_pred, sxy_abaqus).numpy().reshape(-1,1)

        #     ###################################################################################

        #     # # Principal strain - Abaqus
        #     # e1_abaqus = df[df['id']==elem]['ep_1'].values.reshape(-1,1)
        #     # e2_abaqus = df[df['id']==elem]['ep_2'].values.reshape(-1,1)
        
        #     # # Principal stress - Abaqus
        #     # y_princ = torch.from_numpy(df[df['id']==elem][['s1','s2']].values)
        #     # y1_abaqus = y_princ[:,0].numpy().reshape(-1,1)
        #     # y2_abaqus = y_princ[:,1].numpy().reshape(-1,1)

        #     # # Principal stress predictions - ANN
        #     # s1_pred = s_princ[:,idx,0].numpy().reshape(-1,1)
        #     # s2_pred = s_princ[:,idx,1].numpy().reshape(-1,1)

        #     ###################################################################################
        #     #                             RELATIVE ERRORS
        #     ###################################################################################

        #     # mre_s1 = get_re(s1_pred, y1_abaqus).reshape(-1,1)
        #     # mre_s2 = get_re(s2_pred, y2_abaqus).reshape(-1,1)

        #     ###################################################################################

        #     # # Principal strain rate - Abaqus
        #     # de1_abaqus = df[df['id']==elem]['dep_1'].values.reshape(-1,1)
        #     # de2_abaqus = df[df['id']==elem]['dep_2'].values.reshape(-1,1)

        #     # # Principal stress rate - Abaqus
        #     # dy_1 = y[:,idx,0].reshape(-1,1).numpy()
        #     # dy_1 = np.vstack((dy_1,np.array([np.NaN])))
        #     # dy_2 = y[:,idx,1].reshape(-1,1).numpy()
        #     # dy_2 = np.vstack((dy_2,np.array([np.NaN])))

        #     # # Principal stress rate - ANN
        #     # ds1_pred = s_rate[:,idx,0].numpy().reshape(-1,1)
        #     # ds1_pred = np.vstack((ds1_pred,np.array([np.NaN])))
        #     # ds2_pred = s_rate[:,idx,1].numpy().reshape(-1,1)
        #     # ds2_pred = np.vstack((ds2_pred,np.array([np.NaN])))

        #     if tag != last_tag:
        #         print("\n%s\t\tSxx\tSyy\tSxy\tS1\tS2\n" % (tag))
            
        #     print("Elem #%i\t\t%0.3f\t%0.3f\t%0.3f" % (elem, np.mean(mre_sx), np.mean(mre_sy), np.mean(mre_sxy)))

        #     # cols = ['e_xx','e_yy','e_xy','s_xx','s_yy','s_xy','s_xx_pred','s_yy_pred','s_xy_pred',
        #     #         'e_1','e_2','s_1','s_2','s_1_pred','s_2_pred','de_1','de_2','ds_1','ds_2',
        #     #         'dy_1','dy_2','mre_sx','mre_sy','mre_sxy','mre_s1','mre_s2']

        #     res = np.concatenate([ex_abaqus,ey_abaqus,exy_abaqus,sx_abaqus,sy_abaqus,sxy_abaqus,sx_pred,sy_pred,sxy_pred, mre_sx, mre_sy, mre_sxy], axis=1)
            
        #     results = pd.DataFrame(res, columns=cols)
            
        #     results.to_csv(os.path.join(DATA_DIR, f'{tag}_el-{elem}.csv'), header=True, sep=',', float_format='%.6f')

        #     last_tag = tag
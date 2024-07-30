import csv
import constants
import joblib
from functions import (RAFR, GRUModel, GRUModelCholesky, GRUModelJit, GRUModelJitChol)
from mesh_utils import read_mesh
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

from gru_nn import customGRU
from io_funcs import load_config
from plot_utils import plot_lode_triax, plot_rafr, plot_scatter_yield

#-------------------------------------------------------------------------
#                          METHOD DEFINITIONS
#-------------------------------------------------------------------------
def load_pkl(file: str):
    return joblib.load(file)

def scan_ann_files(run: str, dir: str, key: str):

    SCAN_DIR = os.path.join('outputs', dir, 'models', run)
    
    for f in glob.glob(os.path.join(SCAN_DIR, f'*{key}*')):
    
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

def get_field_data(df: pd.DataFrame, vars: dict, pred_vars: dict, n_elems: int, n_tps: int):

    T_STEPS = [round((n_tps-1)*0.5), n_tps-1]
    #T_STEPS = [19,20,21,22,23]

    KEYS = sum([[v for k,v in var.items()] for k_,var in vars.items()],[])

    field_dict = {t: {k: {'abaqus': None, 'ann': None, 'err': None} for k in KEYS} for t in T_STEPS}

    for k, d in vars.items():
        for idx, v_name in d.items():
            if v_name == 'mises':
                s = df[KEYS[:-1]].values.reshape(n_tps,n_elems,-1).transpose(2,1,0)
                s_hat = pred_vars[k].reshape(n_elems, n_tps, -1).transpose(2,0,1)
                x = np.expand_dims(get_mises(*s),-1)
                y = np.expand_dims(get_mises(*s_hat),-1)
            else:
                x = df[v_name].values.reshape(n_tps,n_elems,1).transpose(1,0,2)
                y = pred_vars[k][:,idx].reshape(n_elems,n_tps,1)

            for t, d_ in field_dict.items():
                d_[v_name]['abaqus'] = x[:,t,:]
                d_[v_name]['ann'] = y[:,t,:]
                d_[v_name]['err'] = np.abs(x[:,t,:]-y[:,t,:])

    return field_dict

def import_mesh(dir: str):

    mesh, connectivity, _ = read_mesh(dir)

    nodes = mesh[:,1:]
    connectivity = connectivity[:,1:] - 1

    return nodes, connectivity

def get_abs_err(pred,real):

    '''Calculates the absolute error between a prediction and a real value.'''
    
    abs_err = np.abs(pred-real)

    return abs_err

def get_rmse(pred,real):

    rmse = np.sqrt(np.mean(np.square(pred-real)))

    return rmse

def get_mises(s_x, s_y, s_xy):
    return np.sqrt(np.square(s_x)+np.square(s_y)-s_x*s_y+3*np.square(s_xy))

def dict_to_csv(out_path, file_name, header_template, dict_):
    with open(os.path.join(out_path, file_name), 'w', newline='') as csv_file:  
        writer = csv.writer(csv_file)
        writer.writerow(header_template)
        for key, value in dict_.items():
            if hasattr(value,'items'):
                writer.writerow([key]+[v for _,v in value.items()])
            else:
                writer.writerow([key,value])

def batch_jacobian(y,x, var):
    
    batch = x.size(0)
    inp_dim = x.size(-1)
    out_dim = y.size(-1)

    grad_output = torch.eye(out_dim).unsqueeze(1).repeat(1,batch,1)
    gradient = torch.autograd.grad(y,x,grad_output,retain_graph=True, create_graph=True, is_grads_batched=True)
    J = gradient[0][:,:,-1].permute(1,0,2)
    
    # for i in range(out_dim):
    #     grad_output = torch.zeros([batch,out_dim])
    #     grad_output[:,i] = 1

    #     gradient = torch.autograd.grad(y,x,grad_output,retain_graph=True, create_graph=True)
    #     J[:,i,:] = gradient[0][:,-1]
    #     #print("hey")
    
    return J*(1/var)

def plate_hole_solution(nodes, connectivity, df, pred):

    node_id = np.arange(nodes.shape[0])+1
    mesh = np.hstack([node_id.reshape(-1,1), nodes])

    elem_id = np.arange(connectivity.shape[0])
    connect = np.hstack([elem_id.reshape(-1,1), connectivity])

    left_edge = mesh[mesh[:,1]==0][:,0]-1
    elem_idx = connect[np.any(np.isin(connect[:,1:], left_edge), axis=1)][:,0]

    sx_pred = pred.reshape(connectivity.shape[0],-1,3)[elem_idx,60,0].reshape(-1,1)
    sx_aba = df['sxx_t'].values.reshape(-1,connectivity.shape[0],1)[60, elem_idx]
    cent_y = df['cent_y'].values.reshape(-1,connectivity.shape[0],1)[0, elem_idx]

    r = min(mesh[mesh[:,1]==0][:,-1])
    Tx = df['fxx_t'].values.reshape(-1,connectivity.shape[0])[:,::connectivity.shape[0]][60] / max(mesh[:,1])

    s_ref = Tx * (1 + r ** 2 / (2 * cent_y ** 2) + 3 * r ** 4 / (2 * cent_y ** 4))

    sort_idx = np.argsort(cent_y,0).squeeze(-1)
    
    cent_y = cent_y[sort_idx]
    sx_pred = sx_pred[sort_idx]
    sx_aba = sx_aba[sort_idx]
    s_ref = s_ref[sort_idx]

    return np.concatenate([cent_y, sx_pred, sx_aba, s_ref], axis=1), ['cent_y', 'sx_pred', 'sx_aba', 's_ref']

def save_jit(input_dim, hidden_dim, layer_dim, output_dim, run, proj, cholesky=False, fc_bias=True, gru_bias=True, attention=False):
    if cholesky:
        m = GRUModelJitChol(input_dim=input_dim, hidden_dim=hidden_dim, layer_dim=layer_dim, output_dim=output_dim, fc_bias=fc_bias, gru_bias=gru_bias, attention=attention)
    else:
        m = GRUModelJit(input_dim=input_dim, hidden_dim=hidden_dim, layer_dim=layer_dim, output_dim=output_dim, fc_bias=fc_bias, gru_bias=gru_bias)
    m.load_state_dict(get_ann_model(run, proj))
    m.to(torch.device('cpu')) 
    m.eval()
    traced_model = torch.jit.trace(m,torch.ones(1,4,3).to(torch.device('cpu')))
    traced_model.eval()
    traced_model.save(os.path.join('outputs', proj, 'models', run, f'jit_{run}.pt'))
#--------------------------------------------------------------------------

# Setting Pytorch floating point precision
torch.set_default_dtype(torch.float64)

# Defining ann model to load
#RUN = 'solar-planet-147'
#RUN = 'glowing-tree-54'
#RUN = 'radiant-capybara-3'
#RUN = 'fine-rain-207'
#RUN = 'summer-water-157'
#RUN = 'whole-puddle-134'
#RUN = 'lemon-star-431'
RUN = 'fragrant-2'

if RUN == 'baseline':
    ANN_RUN = 'Baseline'

elif RUN == 'fanciful-water-117':
    ANN_RUN = 'SPD-RNN'

elif RUN == 'helpful-jazz-35':
    ANN_RUN = 'Wp-RNN'

elif RUN == 'sunny-feather-118':
    ANN_RUN = 'SPDw-RNN'
else:
    ANN_RUN = 'RNN'

# Defining output directory
#DIR = 'indirect_crux_gru'
#DIR = 'crux-plastic-jm_sbvf_abs_direct'
#DIR = 'crux-plastic_sbvf_abs_direct'
#DIR = 'sbvfm_indirect_crux_gru'

#PROJ = 'direct_training_gru_moo'
PROJ = 'vfm_training_gru'

# Creating output directories
VAL_DIR = create_dir(dir=RUN, root_dir=os.path.join('outputs', PROJ, 'val'))

# Loading configuration file
config = load_config(scan_ann_files(RUN, PROJ,'config'))

# Importing mesh information
#NODES, CONNECT = import_mesh(config.dirs.mesh)

# # Loading model architecture
# FEATURES, OUTPUTS, INFO, N_UNITS, H_LAYERS, SEQ_LEN = load_file(RUN, DIR, 'arch.pkl')

# Loading data scaler
SCALER_DICT = load_file(RUN, PROJ,'scaler')

ELEMS_VAL = pd.read_csv(os.path.join(VAL_DIR_MULTI,'elems_val.csv'), header=None)[0].to_list()

DRAW_CONTOURS = True
TAGS = ['x15_y15_','s_shaped', 'sigma_shaped', 'd_shaped']
#TAGS = ['s_shaped']
#TAGS = ['s_shaped']

# Setting up ANN model
#model_1 = GRUModelJit(input_dim=len(FEATURES),hidden_dim=N_UNITS,layer_dim=H_LAYERS,output_dim=len(OUTPUTS)+6)

if RUN == 'baseline':
    model_1 = GRUModel(input_dim=len(config.data.inputs),
                    hidden_dim=config.model.hidden_size,
                    layer_dim=config.model.num_layers,
                    output_dim=len(config.data.outputs))
else:
    model_1 = GRUModelCholesky(input_dim=len(config.data.inputs),
                                output_dim=len(config.data.outputs), 
                                hidden_dim=config.model.hidden_size, 
                                layer_dim=config.model.num_layers,
                                cholesky=config.model.cholesky,
                                fc_bias=config.model.bias.fc,
                                gru_bias=config.model.bias.gru,
                                attention=config.model.attention)

#model_1 = customGRU(input_dim=len(FEATURES), hidden_dim=N_UNITS, layer_dim=H_LAYERS, output_dim=len(OUTPUTS), layer_norm=False)
model_1.load_state_dict(get_ann_model(RUN, PROJ))   

save_jit(len(config.data.inputs), config.model.hidden_size, config.model.num_layers, len(config.data.outputs), RUN, PROJ, cholesky=config.model.cholesky, fc_bias=config.model.bias.fc, gru_bias=config.model.bias.gru, attention=config.model.attention)

#Loading validation data
df_list = load_data(dir=VAL_DIR_MULTI, ftype='parquet')
df_list += load_data(dir='data/validation_multi/d_shaped', ftype='parquet')
#df_list += load_data(dir='data/validation_multi/plate_hole', ftype='parquet')
df_list += load_data(dir='data/validation_multi/s_shaped', ftype='parquet')
df_list += load_data(dir='data/validation_multi/sigma_shaped', ftype='parquet')

cols = ['e_xx','e_yy','e_xy','s_xx','s_yy','s_xy','s_xx_pred','s_yy_pred','s_xy_pred',
        'abs_err_sx','abs_err_sy','abs_err_sxy']

stats_rmse = {}
stats_rmse_elems = []

with torch.no_grad():
    last_tag = ''
    for i, df in enumerate(df_list):
        
        # Identifying mechanical test
        tag = df['tag'][0]

        #------------------------------------------------------------------------------------
        #                       CREATING OUTPUT DIRECTORIES
        #------------------------------------------------------------------------------------
        TRIAL_DIR = create_dir(dir=tag,root_dir=VAL_DIR)
        
        DATA_DIR = create_dir(dir='data',root_dir=TRIAL_DIR)

        PLOT_DIR = create_dir(dir='plots',root_dir=TRIAL_DIR)

        if DRAW_CONTOURS:
            COUNTOUR_DIR = create_dir(dir='contour', root_dir=TRIAL_DIR)
        #------------------------------------------------------------------------------------
        #                       SETTING MESH DIRECTORY
        #------------------------------------------------------------------------------------

        if tag == 'plate_hole':
            MESH_DIR = 'data/validation_multi/plate_hole'
        elif tag == 's_shaped':
            MESH_DIR = 'data/validation_multi/s_shaped'
        elif tag == 'sigma_shaped':
            MESH_DIR = 'data/validation_multi/sigma_shaped'
        elif tag == 'd_shaped':
            MESH_DIR = 'data/validation_multi/d_shaped'
        else:
            MESH_DIR = config.dirs.mesh 
        
        # Number of time steps and number of elements
        n_tps = len(list(set(df['t'])))
        n_elems = len(list(set(df['id'])))
        
        X = df[config.data.inputs].values
        y = df[config.data.outputs].values
        
        info = df[config.data.info]

        pad_zeros = torch.zeros((config.data.seq_len-1) * n_elems, X.shape[-1])

        X = torch.cat([pad_zeros, torch.from_numpy(X)], 0)

        if SCALER_DICT['type'] == 'standard':
            X_scaled = (X-SCALER_DICT['mean'])/SCALER_DICT['std']
        elif SCALER_DICT['type'] == 'minmax':
            x_std = (X - SCALER_DICT['stat_vars'][0]) / (SCALER_DICT['stat_vars'][1] - SCALER_DICT['stat_vars'][0])
            X_scaled = x_std * (SCALER_DICT['stat_vars'][2][1] - SCALER_DICT['stat_vars'][2][0]) + SCALER_DICT['stat_vars'][2][0]
        elif SCALER_DICT['type'] == 'preserve_zero':
            X_scaled = X / SCALER_DICT['stat_vars'][0]
        
        x = X_scaled.reshape(n_tps + config.data.seq_len-1, n_elems, -1)
        x = x.unfold(0,config.data.seq_len,1).permute(1,0,3,2)
        x = x.reshape(-1,*x.shape[2:])
        
        y = torch.from_numpy(y)
        y = y.reshape(n_tps,n_elems,-1).permute(1,0,2)
        y = y.reshape(-1,y.shape[-1])

        if tag != 'one_elem':
            t = torch.from_numpy(info['t'].values).reshape(n_tps, n_elems, 1)
        else:
            t = torch.from_numpy(df['t'].dropna())
    
        if config.model.cholesky:
            l = model_1(x)
            s = (l @ (x[:,-1] * SCALER_DICT['std'] + SCALER_DICT['mean']).unsqueeze(-1)).squeeze(-1)
        else:
            s = model_1(x)
            
        data_coord = torch.from_numpy(df[['cent_x', 'cent_y']].values[:n_elems])

        if tag == 'plate_hole':
            
            rafr_obj = RAFR(n_lines=100, line_pts=50, time_stages=4, test=tag)
            
            data = s[:,0].reshape([n_elems,n_tps,1]).permute(1,0,2)

            force_data = torch.from_numpy(df[['fxx_t']].values[::n_elems])
        
        elif tag == 's_shaped' or 'sigma_shaped' or 'd_shaped':

            if tag == 's_shaped':
                rafr_obj = RAFR(n_lines=100, line_pts=50, time_stages=6, test=tag)
            elif tag == 'sigma_shaped':
                rafr_obj = RAFR(n_lines=250, line_pts=100, time_stages=6, test=tag)
            elif tag == 'd_shaped':
                rafr_obj = RAFR(n_lines=200, line_pts=250, time_stages=6, test=tag)

            data = s[:,1].reshape([n_elems,n_tps,1]).permute(1,0,2)

            force_data = torch.from_numpy(df[['fyy_t']].values[::n_elems])

        if tag == 's_shaped' or tag == 'plate_hole' or tag=='sigma_shaped' or tag=='d_shaped':
        
            rafr = rafr_obj.get_rafr(data.float(), data_coord.float(), force_data.float())

            plot_rafr({RUN:rafr}, dir=TRIAL_DIR, specimen=tag)

            joblib.dump(rafr,f'rafr_{tag}_{ANN_RUN}.pkl')

        if tag in TAGS:

            NODES, CONNECT = import_mesh(MESH_DIR)

            plot_scatter_yield(y.reshape(n_elems,n_tps,-1).permute(1,0,2).numpy(), df[config.data.inputs].values.reshape(n_tps, n_elems,-1), df['peeq'].values.reshape(n_tps, n_elems,-1), dir=TRIAL_DIR, test=tag, nodes=NODES, connectivity=CONNECT) 

        #------------------------------------------------------------------------------------
        #                       PREPARING FIELD DATA FOR CONTOUR PLOT
        #------------------------------------------------------------------------------------
        if DRAW_CONTOURS and (tag in TAGS):

            # if tag == 'plate_hole':
            #     MESH_DIR = 'data/validation_multi/plate_hole'
            # elif tag == 's_shaped':
            #     MESH_DIR = 'data/validation_multi/s_shaped'
            # elif tag == 'sigma_shaped':
            #     MESH_DIR = 'data/validation_multi/sigma_shaped'
            # elif tag == 'd_shaped':
            #     MESH_DIR = 'data/validation_multi/d_shaped'
            # else:
            #     MESH_DIR = config.dirs.mesh    
            
            NODES, CONNECT = import_mesh(MESH_DIR)

            if tag == 'plate_hole':
                d, headers = plate_hole_solution(NODES, CONNECT, df, s.numpy())
                d = pd.DataFrame(d, columns=headers)
                d.to_csv(os.path.join(TRIAL_DIR,'plate_solution.csv'), header=True, sep=',',float_format='%.6f', index=False)

            stress_data = torch.cat([y.reshape(n_elems,n_tps,-1)[:,-1].unsqueeze(0), s.reshape(n_elems,n_tps,-1)[:,-1].unsqueeze(0)])
            
            plot_lode_triax(stress_data, TRIAL_DIR)

            vars = {'s': {0: 'sxx_t', 1:'syy_t', 2:'sxy_t', 3:'mises'}}
            
            pred_vars = {'s': s.numpy()}

            fields = get_field_data(df, vars, pred_vars, n_elems, n_tps)

            plot_fields(NODES, CONNECT, fields, out_dir=COUNTOUR_DIR, tag=tag, ann_run=ANN_RUN)

        #------------------------------------------------------------------------------------
        #                              RESHAPING DATA
        #------------------------------------------------------------------------------------

        stats_rmse[tag] = get_rmse(s.numpy(),y.numpy())

        s = s.reshape(n_elems,n_tps,-1)
        y = y.reshape(n_elems,n_tps,-1)

        elems_dict = {elem: {'sx': None, 'sy': None, 'sxy': None} for elem in ELEMS_VAL}

        for elem in ELEMS_VAL:
            idx = elem - 1
            # Strain values - Abaqus
            ex_abaqus = df[df['id']==elem]['exx_t'].values.reshape(-1,1)
            ey_abaqus = df[df['id']==elem]['eyy_t'].values.reshape(-1,1)
            exy_abaqus = df[df['id']==elem]['exy_t'].values.reshape(-1,1)

            # Stress values - Abaqus
            sx_abaqus = df[df['id']==elem]['sxx_t'].values.reshape(-1,1)
            sy_abaqus = df[df['id']==elem]['syy_t'].values.reshape(-1,1)
            sxy_abaqus = df[df['id']==elem]['sxy_t'].values.reshape(-1,1)

            # Stress predictions - ANN
            sx_pred = s[idx,:,0].reshape(-1,1)
            sy_pred = s[idx,:,1].reshape(-1,1)
            sxy_pred= s[idx,:,2].reshape(-1,1)

            s_0_pred = s[idx,0]
            s_0 = y[idx,0]

            ###################################################################################
            #                             RELATIVE ERRORS
            ###################################################################################

            abs_err_sx = get_abs_err(sx_pred, sx_abaqus).numpy().reshape(-1,1)
            abs_err_sy = get_abs_err(sy_pred, sy_abaqus).numpy().reshape(-1,1)
            abs_err_sxy = get_abs_err(sxy_pred, sxy_abaqus).numpy().reshape(-1,1)

            rmse_sx = get_rmse(sx_pred.numpy(), sx_abaqus)
            rmse_sy = get_rmse(sy_pred.numpy(), sy_abaqus)
            rmse_sxy = get_rmse(sxy_pred.numpy(), sxy_abaqus)
            rmse_s0 = get_rmse(s_0_pred.numpy(), s_0.numpy())

            stats_rmse_elems.append([elem,rmse_sx,rmse_sy,rmse_sxy, rmse_s0])
            # elems_dict[elem]['sx'] = rmse_sx
            # elems_dict[elem]['sy'] = rmse_sy
            # elems_dict[elem]['sxy'] = rmse_sxy    

            if tag != last_tag:
                print("\n%s\t\tSxx\tSyy\tSxy\tS_0\n" % (tag))
            
            print("Elem #%i\t\t%0.3f\t%0.3f\t%0.3f\t%0.3f" % (elem, rmse_sx, rmse_sy, rmse_sxy, rmse_s0))

            res = np.concatenate([ex_abaqus,ey_abaqus,exy_abaqus,sx_abaqus,sy_abaqus,sxy_abaqus,sx_pred,sy_pred,sxy_pred, abs_err_sx, abs_err_sy, abs_err_sxy], axis=1)
            
            results = pd.DataFrame(res, columns=cols)
            
            results.to_csv(os.path.join(DATA_DIR, f'{tag}_el-{elem}.csv'), header=True, sep=',', float_format='%.10f')

            last_tag = tag
    
    #dict_to_csv(VAL_DIR, 'stats_elems.csv', ['tag','elem','rmse_sx','rmse_sy','rmse_sxy'], stats_rmse_elems)

a = pd.DataFrame(stats_rmse_elems,columns=['elem','rmse_sx','rmse_sy','rmse_sxy','rmse_s0'])
a.to_csv(os.path.join(VAL_DIR,'stats_elems.csv'), header=True, sep=',',float_format='%.6f', index=False) 

dict_to_csv(VAL_DIR,'stats.csv',['Run','RMSE'],stats_rmse)        
# with open(os.path.join(VAL_DIR,'stats.csv'), 'w', newline='') as csv_file:  
#     writer = csv.writer(csv_file)
#     writer.writerow(['Run','RMSE'])
#     for key, value in stats_rmse.items():
#         writer.writerow([key, value])
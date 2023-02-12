
import constants
import joblib
from functions import (load_dataframes, rotate_tensor, select_features_multi, NeuralNetwork)
import matplotlib.pyplot as plt
import pandas as pd
from cycler import cycler
import shap
import torchmetrics
import torch
import random
import itertools
import numpy as np
import math
from torch.autograd import Variable
from matplotlib.gridspec import GridSpec
import wandb
import os
from constants import *

def get_mre(pred,real):
    
    mre = np.abs(pred-real)[1:]/real[1:]
    mre = np.vstack(([[0.0]],mre))

    return mre


def get_elastic_matrix(mean,var):
    s = np.sqrt(var)
    D = np.array([[230769.231,69230.769,0],[69230.769,230769.231,0],[0,0,80769.231]])
    w = D*s
    b = -w@(-mean/s)
    return torch.from_numpy(w),torch.from_numpy(b)

def shap_analysis(model, df_list, scaler):

    f = lambda x: model(Variable(torch.from_numpy(x))).detach().numpy()

    # Merging training data
    data = pd.concat(df_list, axis=0, ignore_index=True)

    # Reorganizing dataset by time increment, subsequent grouping by tag and final shuffling
    data_by_t = [df for _, df in data.groupby(['t'])]
    random.shuffle(data_by_t)
    data_by_tag = [[df for _, df in group.groupby(['tag'])] for group in data_by_t]
    random.shuffle(data_by_tag)
    data_by_batches = list(itertools.chain(*data_by_tag))
    random.shuffle(data_by_batches)

    data = pd.concat(data_by_batches).reset_index(drop=True)

    batches = np.array_split(data.index.tolist(),len(data_by_batches))
    idx = list(range(len(batches)))
    test_batch_idxs = random.sample(idx, math.floor(len(batches)*0.3))
    test_batch_idxs.sort()
    train_batch_idxs = list(set(idx).difference(test_batch_idxs))

    train_idx = list(itertools.chain.from_iterable([batches[i].tolist() for i in train_batch_idxs]))
    test_idx = list(itertools.chain.from_iterable([batches[i].tolist() for i in test_batch_idxs]))

    X, _, _, _, _ = select_features_multi(data)

    X = scaler.transform(X)

    model.eval()
    with torch.no_grad():
        explainer = shap.KernelExplainer(f,X[test_idx[:64],:])
        shap_values = explainer.shap_values(X[test_idx[64:],:], nsamples='auto')
    
    fig = plt.figure()
    fig.suptitle('SHAP Analysis')
    fig.set_size_inches(16, 16, forward=True)
    fig.subplots_adjust(wspace=0.6)
    fig.tight_layout()
    g_s = GridSpec(1, X.shape[1], figure=fig)

    o_names = [r'$\sigma_{xx}$',r'$\sigma_{yy}$',r'$\sigma_{xy}$']
    f_names = [r'$\varepsilon_{xx}$',r'$\varepsilon_{yy}$',r'$\varepsilon_{xy}$']
    for i,feature in enumerate(shap_values):
        ax = fig.add_subplot(g_s[0,i])
        shap.summary_plot(shap_values[i], X[test_idx[64:],:],feature_names=f_names,show=False)
        ax.set_title(o_names[i])
        ax.set_box_aspect(1)
        cb=fig.axes[2*i+1]
        cb.set_ylabel('')
        cb.set_box_aspect(3)
    plt.show()
    return shap_values

plt.rcParams.update(constants.PARAMS)

default_cycler = (cycler(color=["#ef476f","#118ab2","#073b4c"]))

plt.rc('axes', prop_cycle=default_cycler)

torch.set_default_dtype(torch.float64)

# Loading data
df_list, file_names = load_dataframes(os.path.join(constants.VAL_DIR_MULTI,'processed_v'), preproc=False)

file_names = [file_name.split('/')[-1] for file_name in file_names]
# Loading data scalers
#x_scaler = joblib.load('outputs/9-elem-200-elastic_testfull/models/[6-4x1-3]-9-elem-200-elastic-4-VFs-scaler_x.pkl')
#x_scaler = joblib.load('outputs/9-elem-200-plastic_testfull/models/[6-8x1-3]-9-elem-200-plastic-6-VFs-scaler_x.pkl')
#x_scaler = joblib.load('outputs/9-elem-1000-elastic_indirect/models/[3-6x1-3]-9-elem-1000-elastic-4-VFs-scaler_x.pkl')

#x_scaler = joblib.load('outputs/9-elem-50-elastic_sbvf_abs/models/[3-3x0-3]-9-elem-50-elastic-12-VFs-scaler_x.pkl')
#y_scaler = joblib.load('outputs/9-elem-1000-elastic_indirect/models/[3-3x1-3]-9-elem-1000-elastic-scaler_y.pkl')


INPUTS = 6
OUTPUTS = 2
N_UNITS = [20,20,20,20]
H_LAYERS = len(N_UNITS)

# Loading ANN model
model_1 = NeuralNetwork(INPUTS, OUTPUTS, N_UNITS, len(N_UNITS))
#model_1 = InputConvexNN(6, 3, [9,9,9], 3)
# model_2 = NeuralNetwork(3, 1, 8, 1)
# model_3 = NeuralNetwork(3, 1, 8, 1)
#model_1.load_state_dict(torch.load('outputs/9-elem-200-elastic_testfull/models/[6-4x1-3]-9-elem-200-elastic-4-VFs.pt'))
#model_1.load_state_dict(torch.load('outputs/9-elem-200-plastic_testfull/models/[6-8x1-3]-9-elem-200-plastic-6-VFs_1.pt'))

RUN = 'major-eon-1'
DIR = 'crux-plastic_sbvf_abs_direct'

for r, d, f in os.walk(f'outputs/{DIR}'):
    for file in f:
        if RUN in file and '.pt' in file:
            model_1.load_state_dict(torch.load(f'outputs/{DIR}/models/{file}'))
        
        if RUN in file and 'scaler_x.pkl' in file:
            std,mean = joblib.load(f'outputs/{DIR}/models/{file}')

try:
    os.makedirs(f'outputs/{DIR}/val/{RUN}')
except FileExistsError:
    pass
#std, mean =  joblib.load(os.path.join(TRAIN_MULTI_DIR,'scaler_x.pkl'))
            

# x_scaler = joblib.load(f'outputs/{DIR}/models/{RUN}-[{INPUTS}-{N_UNITS[0]}x{H_LAYERS}-{OUTPUTS}]-crux-plastic-1022-VFs-scaler_x.pkl')
# #model_1.load_state_dict(torch.load('outputs/9-elem-50-elastic_sbvf_abs/models/[3-3x0-3]-9-elem-50-elastic-12-VFs.pt'))
# model_1.load_state_dict(torch.load(f'outputs/crux-plastic_sbvf_abs_direct/models/{RUN}-[{INPUTS}-{N_UNITS[0]}x{H_LAYERS}-{OUTPUTS}]-crux-plastic-1022-VFs.pt'))

# model_2 = NeuralNetwork(3, 3, 0, 0)

# w,b = get_elastic_matrix(x_scaler.mean_,x_scaler.var_)
# # w = torch.tensor([[79.12783467,8.415587345,0],[23.7383503,28.05195794,0],[0,0,5.06826111]],dtype=torch.float64)
# # b = torch.tensor([1.04E+02,-6.10646E-07,5.425392226],dtype=torch.float64)
# model_2.layers[0].weight = torch.nn.Parameter(w)
# model_2.layers[0].bias = torch.nn.Parameter(b)

model_1.eval()
#model_2.eval()
#a = shap_analysis(model_1,df_list,x_scaler)

# Sampling data pass random seed for random sampling
#sampled_dfs = data_sampling(df_list, constants.DATA_SAMPLES)

err = torchmetrics.MeanAbsoluteError()
r2 = torchmetrics.R2Score()

elem_list = pd.read_csv(os.path.join(VAL_DIR_MULTI,'elems_val.csv'), header=None)[0].to_list()
r2_scores = []
mres = []

# entity = 'rmbl'
# project='indirect_training_tests'

# api = wandb.Api()
# # # # Start a W&B run to log data
# DISPLAY_NAME = '2v5q79tx'
# run = api.run(f'{entity}/{project}/{DISPLAY_NAME}')

# run_ = wandb.init(entity=entity, project=project, id=run.id, resume="must")
# # # wandb.artifactsp
# generate_run = wandb.init(entity=entity,project=project,job_type='generate_artifacts')

# artifact = wandb.Artifact('results', type='image')

print("--------------------------------------\n\tMean Absolute Error\n--------------------------------------")
with torch.no_grad():
    last_tag = ''
    for i, df in enumerate(df_list):
        
        if df['tag'].iloc[0] != last_tag:
            print("\n%s\t\tSxx\tSyy\tSxy\tS1\tS2\n" %(df['tag'].iloc[0]))
            last_tag = df['tag'].iloc[0]
        
        X, y, _, _, info = select_features_multi(df)
        #X_scaled = torch.from_numpy(X)
        X_scaled = (torch.from_numpy(X.values) - mean) / std
        #X_scaled=torch.from_numpy(x_scaler.transform(X))
        X_scaled = X_scaled[~torch.any(X_scaled.isnan(),dim=1)]
       
        y = torch.from_numpy(y.values)
        y = y[~torch.any(y.isnan(),dim=1)]

        theta_ep = torch.from_numpy(info[['theta_ep']].values)
        theta_sp = torch.from_numpy(info[['theta_sp']].values)

        t_pts = len(set(info['inc'].values))
        n_elems = len(info['tag'])//128

        t = torch.from_numpy(info['t'].values).reshape(t_pts,n_elems,1)
        dt = torch.diff(t,dim=0).reshape(-1,1)
        #dt = torch.cat((torch.as_tensor([[0]]),torch.diff(t,dim=0)),0)
 
        s_rate = model_1(X_scaled) # stress rate.
        
        s_princ = torch.zeros(t.shape[0],s_rate.shape[-1])
        #s_princ = s_rate*dt.repeat(1,2)
        s_princ[1:,:] = s_rate*dt.repeat(1,2)
        s_princ = torch.cumsum(s_princ,0)

        #s_princ = torch.cumulative_trapezoid(s_rate,t,dim=0)
        #s_princ = torch.cat((torch.as_tensor([[0,0]]),s_princ),0)

        s_princ_mat = torch.zeros([s_princ.shape[0],2,2])        
        s_princ_mat[:,0,0] = s_princ[:,0]
        s_princ_mat[:,1,1] = s_princ[:,1]

        s = rotate_tensor(s_princ_mat.numpy(),theta_sp.reshape(-1).numpy(),is_reverse=True)
        s = s[:,[0,1,1],[0,1,0]]

        # L = torch.zeros([51, 3, 3])
        # tril_indices = torch.tril_indices(row=3, col=3, offset=0)
        # L[:, tril_indices[0], tril_indices[1]] = y_pred_inv[:]
        # H = L @torch.transpose(L,1,2)

        # d_e= torch.from_numpy(info[['d_exx','d_eyy','d_exy']].values).reshape([51,3,1])
        # d_s = (H @ d_e).squeeze()
        # y_pred_inv = torch.cumsum(d_s,0).detach().numpy()
       
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

        mre_sx = get_mre(sx_pred, sx_abaqus).reshape(-1,1)
        mre_sy = get_mre(sy_pred, sy_abaqus).reshape(-1,1)
        mre_sxy = get_mre(sxy_pred, sxy_abaqus).reshape(-1,1)

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

        mre_s1 = get_mre(s1_pred, y1_abaqus).reshape(-1,1)
        mre_s2 = get_mre(s2_pred, y2_abaqus).reshape(-1,1)

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

        elem_list.append(df['id'].iloc[0])

        print("Elem #%i\t\t\t%0.3f\t%0.3f\t%0.3f\t%0.3f\t%0.3f" % (df['id'].iloc[0], np.mean(mre_sx), np.mean(mre_sy), np.mean(mre_sxy), np.mean(mre_s1), np.mean(mre_s2)))

        cols = ['e_xx','e_yy','e_xy','s_xx','s_yy','s_xy','s_xx_pred','s_yy_pred','s_xy_pred','e_1','e_2','s_1','s_2','s_1_pred','s_2_pred','de_1','de_2','ds_1','ds_2','dy_1','dy_2','mre_sx','mre_sy','mre_sxy','mre_s1','mre_s2']

        if df['id'].iloc[0] in elem_list:
            
            res = np.concatenate([ex_abaqus,ey_abaqus,exy_abaqus,sx_abaqus,sy_abaqus,sxy_abaqus,sx_pred,sy_pred,sxy_pred,e1_abaqus,e2_abaqus,y1_abaqus,y2_abaqus,s1_pred,s2_pred,de1_abaqus,de2_abaqus,ds1_pred,ds2_pred,dy_1,dy_2,mre_sx, mre_sy, mre_sxy, mre_s1, mre_s2], axis=1)
            
            results = pd.DataFrame(res, columns=cols)
            
            results.to_csv(f'outputs/{DIR}/val/{RUN}/' + df['tag'].iloc[0]+'_'+str(df['id'].iloc[0])+'.csv', header=True, sep=',', float_format='%.6f')
            
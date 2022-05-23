from distutils.log import error
from sklearn.metrics import r2_score
from tensorflow.python.keras.backend import dtype
import constants
import joblib
from functions import (ICNN, load_dataframes, select_features_multi,NeuralNetwork)
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
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
    #plt.show()
    return shap_values

plt.rcParams.update(constants.PARAMS)

default_cycler = (cycler(color=["#ef476f","#118ab2","#073b4c"]))

plt.rc('axes', prop_cycle=default_cycler)

torch.set_default_dtype(torch.float64)

# Loading data
df_list, file_names = load_dataframes(constants.VAL_DIR_MULTI)

file_names = [file_name.split('/')[-1] for file_name in file_names]
# Loading data scalers
#x_scaler = joblib.load('outputs/9-elem-200-elastic_testfull/models/[6-4x1-3]-9-elem-200-elastic-4-VFs-scaler_x.pkl')
#x_scaler = joblib.load('outputs/9-elem-200-plastic_testfull/models/[6-8x1-3]-9-elem-200-plastic-6-VFs-scaler_x.pkl')
#x_scaler = joblib.load('outputs/9-elem-1000-elastic_indirect/models/[3-6x1-3]-9-elem-1000-elastic-4-VFs-scaler_x.pkl')

x_scaler = joblib.load('outputs/9-elem-50-elastic_sbvf/models/[3-3x0-3]-100-elem-25-elastic-12-VFs-scaler_x.pkl')
#y_scaler = joblib.load('outputs/9-elem-1000-elastic_indirect/models/[3-3x1-3]-9-elem-1000-elastic-scaler_y.pkl')

#x_scaler = joblib.load('outputs/9-elem-50-plastic_sbvf/models/[6-3x1-3]-9-elem-50-plastic-51-VFs-scaler_x.pkl')


# Loading ANN model
model_1 = NeuralNetwork(3, 3, 0, 0)
# model_2 = NeuralNetwork(3, 1, 8, 1)
# model_3 = NeuralNetwork(3, 1, 8, 1)
#model_1.load_state_dict(torch.load('outputs/9-elem-200-elastic_testfull/models/[6-4x1-3]-9-elem-200-elastic-4-VFs.pt'))
#model_1.load_state_dict(torch.load('outputs/9-elem-200-plastic_testfull/models/[6-8x1-3]-9-elem-200-plastic-6-VFs_1.pt'))

model_1.load_state_dict(torch.load('outputs/9-elem-50-elastic_sbvf/models/[3-3x0-3]-100-elem-25-elastic-12-VFs_1.pt'))
#model_1.load_state_dict(torch.load('outputs/9-elem-50-plastic_sbvf/models/[6-3x1-3]-9-elem-50-plastic-51-VFs.pt'))
model_1.eval()

#a = shap_analysis(model_1,df_list,x_scaler)

# Sampling data pass random seed for random sampling
#sampled_dfs = data_sampling(df_list, constants.DATA_SAMPLES)

err = torchmetrics.MeanAbsoluteError()
r2 = torchmetrics.R2Score()

elem_list = []
r2_scores = []
maes = []

print("--------------------------------------\n\tMean Absolute Error\n--------------------------------------")
with torch.no_grad():
    for i, df in enumerate(df_list):
        
        if df['id'][0] == 1:
            print("\n%s\tSxx\tSyy\tSxy\tr2_x\tr2_y\tr2_xy\n" %(df['tag'][0]))
        
        X, y, _, _, _ = select_features_multi(df)
        X_scaled=x_scaler.transform(X)
        
        y_pred_inv = model_1(torch.tensor(X_scaled)).detach().numpy()
    
        ex_var_abaqus = df['exx_t']
        ey_var_abaqus = df['eyy_t']
        exy_var_abaqus = df['exy_t']
        sx_var_abaqus = df['sxx_t']
        sy_var_abaqus = df['syy_t']
        sxy_var_abaqus = df['sxy_t']

        sx_pred_var = y_pred_inv[:,0]
        sy_pred_var = y_pred_inv[:,1]
        sxy_pred_var = y_pred_inv[:,2]

        mse_x = err(torch.from_numpy(sx_pred_var), torch.from_numpy(sx_var_abaqus.values))
        mse_y = err(torch.from_numpy(sy_pred_var), torch.from_numpy(sy_var_abaqus.values))
        mse_xy = err(torch.from_numpy(sxy_pred_var), torch.from_numpy(sxy_var_abaqus.values))

        maes.append([mse_x, mse_y, mse_xy])

        r2_x = r2(torch.from_numpy(sx_pred_var), torch.from_numpy(sx_var_abaqus.values))
        r2_y = r2(torch.from_numpy(sy_pred_var), torch.from_numpy(sy_var_abaqus.values))
        r2_xy = r2(torch.from_numpy(sxy_pred_var), torch.from_numpy(sxy_var_abaqus.values))

        r2_scores.append([r2_x, r2_y, r2_xy])

        elem_list.append(df['id'][0])

        print("Elem #%i\t\t%0.3f\t%0.3f\t%0.3f\t%0.3f\t%0.3f\t%0.3f" % (df['id'][0], mse_x, mse_y, mse_xy, r2_x, r2_y, r2_xy))
       
        # fig , (ax1, ax2, ax3) = plt.subplots(1,3)
        # fig.suptitle(r''+ df['tag'][0].replace('_','\_') + ': element \#' + str(df['id'][0]),fontsize=14)
        # fig.set_size_inches(10, 5)
        # fig.subplots_adjust(bottom=0.2, top=0.8)
    
        # ax1.plot(ex_var_abaqus, sx_var_abaqus, label='ABAQUS')
        # ax1.plot(ex_var_abaqus, sx_pred_var, label='ANN')
        # ax1.set(xlabel=r'$\varepsilon_{xx}$', ylabel=r'$\sigma_{xx}$ [MPa]')
        # #ax1.set_title(r'$\text{MSE}=%0.3f$' % (mse_x), fontsize=11)
        # ax2.plot(ey_var_abaqus, sy_var_abaqus, label='ABAQUS')
        # ax2.plot(ey_var_abaqus, sy_pred_var, label='ANN')
        # ax2.set(xlabel=r'$\varepsilon_{yy}$', ylabel=r'$\sigma_{yy}$ [MPa]')
        # ax2.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
        # #ax2.set_title(r'$\text{MSE}=%0.3f$' % (mse_y), fontsize=11)
        # ax3.plot(exy_var_abaqus, sxy_var_abaqus, label='ABAQUS')
        # # ax3.plot(exy_var_abaqus, func(exy_var_abaqus, *popt), label='ABAQUS')
        # ax3.plot(exy_var_abaqus, sxy_pred_var, label='ANN')
        # ax3.set(xlabel=r'$\varepsilon_{xy}$', ylabel=r'$\tau_{xy}$ [MPa]')
        # ax3.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
        # #ax3.set_title(r'$\text{MSE}=%0.3f$' % (mse_xy), fontsize=11)
        # handles, labels = ax3.get_legend_handles_labels()
        # fig.legend(handles, labels, loc='lower center',ncol=2)
        
        #plt.show()
        
        if df['id'][0] in elem_list:
            predictions = pd.DataFrame(y_pred_inv, columns=['pred_x','pred_y','pred_xy'])
            #stats = pd.DataFrame([maes, r2_scores], columns=['mae_x','mae_y','mae_xy','r2_x','r2_y','r2_xy'])
            results = pd.concat([df[['exx_t','eyy_t','exy_t','sxx_t','syy_t','sxy_t']],predictions], axis=1)
            results.to_csv('outputs/9-elem-50-plastic_sbvf/val/' + df['tag'][0]+'_'+str(df['id'][0])+'.csv', header=True, sep=',',float_format='%.12f')
            

        
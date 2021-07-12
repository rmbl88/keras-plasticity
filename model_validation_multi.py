import constants
import joblib
import tensorflow as tf
from tensorflow import keras
from functions import custom_loss, load_dataframes, data_sampling, select_features, select_features_multi, standardize_data
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from cycler import cycler
from operator import itemgetter

plt.rcParams.update(constants.PARAMS)

default_cycler = (cycler(color=["#ef476f","#118ab2","#073b4c"]))

plt.rc('axes', prop_cycle=default_cycler)

# Loading data
df_list, file_names = load_dataframes(constants.VAL_DIR_MULTI)

# Loading data scalers
x_scaler, y_scaler = joblib.load('models/ann3/scalers.pkl')

# Loading ANN model
model = keras.models.load_model('models/ann3', compile=False)

model.summary()

# Sampling data pass random seed for random sampling
sampled_dfs = data_sampling(df_list, constants.DATA_SAMPLES)

for i, df in enumerate(sampled_dfs):

    #noise = np.random.normal(0, 0.1, list(df.shape))

    X, y = select_features_multi(df)

    y = y.drop(['fxx_t', 'fyy_t', 'fxy_t'], axis=1)

    # Apply previous validation dataset
    X_val, y_val, _, _ = standardize_data(X, y, x_scaler, y_scaler)

    y_pred = model.predict(X_val)

    y_pred_inv = y_scaler.inverse_transform(y_pred)

    ex_var_abaqus = df['exx_t']
    ey_var_abaqus = df['eyy_t']
    exy_var_abaqus = df['exy_t']
    sx_var_abaqus = df['sxx_t']
    sy_var_abaqus = df['syy_t']
    sxy_var_abaqus = df['sxy_t']

    sx_pred_var = y_pred_inv[:,0]
    sy_pred_var = y_pred_inv[:,1]
    sxy_pred_var = y_pred_inv[:,2]
    
    fig , (ax1, ax2, ax3) = plt.subplots(1,3)
    ax1.plot(ex_var_abaqus, sx_var_abaqus, label='ABAQUS')
    ax1.plot(ex_var_abaqus, sx_pred_var, label='ANN')
    ax1.set(xlabel=r'$\varepsilon$', ylabel=r'$\sigma_{xx}$ [MPa]')
    ax2.plot(ey_var_abaqus, sy_var_abaqus, label='ABAQUS')
    ax2.plot(ey_var_abaqus, sy_pred_var, label='ANN')
    ax2.set(xlabel=r'$\varepsilon$', ylabel=r'$\sigma_{yy}$ [MPa]')
    ax3.plot(exy_var_abaqus, sxy_var_abaqus, label='ABAQUS')
    ax3.plot(exy_var_abaqus, sxy_pred_var, label='ANN')
    ax3.set(xlabel=r'$\varepsilon$', ylabel=r'$\tau_{xy}$ [MPa]')
    handles, labels = ax3.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center')

    #results = model.evaluate(X_val,y_val[:,:3])
    
plt.show()
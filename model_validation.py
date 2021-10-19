import constants
import joblib
import tensorflow as tf
from tensorflow import keras
from functions import load_dataframes, data_sampling, select_features, standardize_data
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from cycler import cycler
from operator import itemgetter
import matplotlib.animation as animation

def update(num, x_1, y_1, x_2, y_2, line_1, line_2):
    line_1.set_data(x_1[:num], y_1[:num])
    line_2.set_data(x_2[:num], y_2[:num])
    #line.axes.axis([0, 10, 0, 1])
    return line_1, line_2

plt.rcParams.update(constants.PARAMS)

my_writer=animation.PillowWriter(fps=30, codec='libx264', bitrate=2)
#my_writer = animation.FFMpegWriter(fps=30)

default_cycler = (cycler(color=["#ef476f","#118ab2","#073b4c"]))

plt.rc('axes', prop_cycle=default_cycler)

# Loading data
df_list, file_names = load_dataframes(constants.VAL_DIR)

# Loading data scalers
x_scaler, y_scaler = joblib.load('models/ann1/scalers.pkl')

# Loading ANN model
model = keras.models.load_model('models/ann1')

model.summary()

# Sampling data pass random seed for random sampling
sampled_dfs = data_sampling(df_list, constants.DATA_SAMPLES)

for i, df in enumerate(sampled_dfs):

    #noise = np.random.normal(0, 0.1, list(df.shape))

    X, y = select_features(df)

    # Apply previous validation dataset
    X_val, y_val, _, _ = standardize_data(X, y, x_scaler, y_scaler)

    y_pred = model.predict(X_val)

    y_pred_inv = y_scaler.inverse_transform(y_pred)

    file_name = file_names[i].split('/')[-1]

    if 'U1' in file_name:
        x_var_abaqus = df['exx_t']
        y_var_abaqus = df['sxx_t']
        y_pred_var = y_pred_inv[:,0]
        y_label = r'$\sigma_{xx}$ [MPa]'

    elif 'U2' in file_name:
        x_var_abaqus = df['eyy_t']
        y_var_abaqus = df['syy_t']
        y_pred_var = y_pred_inv[:,1]
        y_label = r'$\sigma_{yy}$ [MPa]'
    
    elif 'shear' in file_name:
        x_var_abaqus = df['exy_t']
        y_var_abaqus = df['sxy_t']
        y_pred_var = y_pred_inv[:,2]
        y_label = r'$\tau_{xy}$ [MPa]'
    
    # plt.figure(i,tight_layout='inches')
    # plt.xlabel(r'$\varepsilon$')
    # plt.ylabel(y_label)
    # plt.plot(x_var_abaqus, y_var_abaqus, label='ABAQUS')
    # plt.plot(x_var_abaqus, y_pred_var, '--', label='ANN')
    # plt.legend(loc='lower center', bbox_to_anchor=(0.47,-0.25), ncol=2)

    fig = plt.figure(i,tight_layout='inches',figsize=(7,6)) 
    ax = fig.add_subplot(111)
    
    line_1, = ax.plot(x_var_abaqus, y_var_abaqus, label='ABAQUS')
    line_2, = ax.plot(x_var_abaqus, y_pred_var, '--', label='ANN')
    ax.set_xlabel(r'$\varepsilon$')
    ax.set_ylabel(y_label)
    plt.legend(loc='lower center', bbox_to_anchor=(0.47,-0.25), ncol=2)

    #results = model.evaluate(X_val,y_val)

    ani = animation.FuncAnimation(fig, update, len(x_var_abaqus), fargs=[x_var_abaqus, y_var_abaqus, x_var_abaqus, y_pred_var, line_1, line_2], interval=30, blit=False)
    ani.save('prints/'+str(i)+'.gif', writer=my_writer, dpi=300)
    #plt.show()
    
#plt.show()
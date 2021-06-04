import constants
import joblib
import tensorflow as tf
from tensorflow import keras
from functions import load_dataframes, data_sampling, select_features, standardize_data
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update(constants.PARAMS)

# Loading data
df_list = load_dataframes(constants.VAL_DIR)

# Loading data scalers
x_scaler, y_scaler = joblib.load('models/ann1/scalers.pkl')

# Loading ANN model
model = keras.models.load_model('models/ann1')

# Sampling data pass random seed for random sampling
sampled_dfs = data_sampling(df_list, constants.DATA_SAMPLES)

for i, df in enumerate(sampled_dfs):

    X, y = select_features(df)

    # Apply previous validation dataset
    X_val, y_val, _, _ = standardize_data(X, y, x_scaler, y_scaler)

    y_pred = model.predict(X_val)

    y_pred_inv = y_scaler.inverse_transform(y_pred)

    plt.figure(i)
    plt.xlabel(r'$\varepsilon$')
    plt.ylabel(r'$\sigma$ [MPa]')
    plt.plot(df['eyy_t+dt'], df['syy_t+dt'], label='ABAQUS')
    plt.plot(df['eyy_t+dt'], y_pred_inv[:,1], label='ANN')

    results = model.evaluate(X_val,y_val)
    
plt.show()

print('hey')
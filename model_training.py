# %%
from sklearn.utils import shuffle
from re import S
from functions import data_sampling, load_dataframes, select_features, standardize_data, plot_history
from tensorflow import keras
from sklearn.model_selection import train_test_split
import tensorflow as tf
import kerastuner as kt
import matplotlib.pyplot as plt
import pandas as pd
import constants
import joblib 

def create_model(hp):
    
    # Initializing sequential model
    model = keras.Sequential()

    # Defining input layer    
    model.add(keras.layers.Input(shape=(6,)))

    # Tuning the number of hidden layers and hidden units
    for i in range(hp.Int('num_layers', 1, 1)):

        hp_units = hp.Int('units_' + str(i), min_value=5, max_value=50, step=5)
        hp_activation = hp.Choice('activ_' + str(i), values=['relu','tanh','selu','elu'])
        hp_weight_init = hp.Choice('init_' + str(i), values=['truncated_normal', 'variance_scaling', 'orthogonal'])
        hp_l2 = hp.Choice('l2_reg_' + str(i),values=[1e-1, 1e-2, 1e-3, 1e-4])
        
        model.add(keras.layers.Dense(units=hp_units,
                                    activation=hp_activation,
                                    use_bias=True,
                                    bias_initializer='ones',
                                    kernel_initializer=hp_weight_init
                                    #kernel_regularizer=keras.regularizers.L2(l2=hp_l2)
                                    )
        )
    
    # Defining the output layer
    model.add(keras.layers.Dense(units=3))

    # Tune the learning rate for the optimizer
    # Choose an optimal value from 0.01, 0.001, or 0.0001
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4, 1e-5])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.005),
                  loss=keras.losses.MeanSquaredError(),
                  metrics=['mae', 'mse'])

    return model

# Loading data
df_list = load_dataframes(constants.TRAIN_DIR)

# Sampling data pass random seed for random sampling
sampled_dfs = data_sampling(df_list, constants.DATA_SAMPLES)

# for i, df in enumerate(sampled_dfs):
#     plt.plot(df['eyy_t+dt'], df['syy_t+dt'], 'go--', linewidth=0.15, markersize=0.75)
    
# plt.show()

# Merging training data
data = pd.concat(sampled_dfs, axis=0, ignore_index=True)

X, y = select_features(data)

# Shuffling dataset
X_shuf, y_shuf = shuffle(X, y, random_state=constants.SEED)

# Splitting data into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X_shuf, y_shuf, test_size=constants.TEST_SIZE, random_state=constants.SEED)

# Normalizing/Standardizing training dataset
X_train, y_train, x_scaler, y_scaler = standardize_data(X_train, y_train)

# Apply previous scaling to test dataset
X_test, y_test, _, _ = standardize_data(X_test, y_test, x_scaler, y_scaler)

# Defining tuner
tuner = kt.Hyperband(create_model,
                     objective='mae',
                     max_epochs=75,
                     factor=3,
                     executions_per_trial=2,
                     seed=constants.SEED,
                     directory='hyperband',
                     project_name='hyperparameter_tuning')

stop_early = tf.keras.callbacks.EarlyStopping(monitor='mse', patience=20)

# Performing hyperparameter search
tuner.search(X_train, y_train, epochs=100, validation_data=(X_test, y_test), shuffle=True)

# Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""\nThe hyperparameter search is complete:\n
- Optimal number of hidden layers: {best_hps.get('num_layers')}
""")

for i in range(best_hps.get('num_layers')):
    print('\tNeurons in hidden layer ' + str(i+1) + ': ' + str(best_hps.get('units_' + str(i))))
    print('\t\t- Activation: ' + str(best_hps.get('activ_' + str(i))))
    print('\t\t- L2 regularization: ' + str(best_hps.get('l2_reg_' + str(i))))

print(f"""
- Optimal learning rate for the optimizer: {best_hps.get('learning_rate')}
""")

#Build the model with the optimal hyperparameters and train it on the data
# model = tuner.hypermodel.build(best_hps)
# print(model.summary())
# history = model.fit(X_train, y_train, epochs=200, validation_data=(X_test, y_test), shuffle=True)

# val_acc_per_epoch = history.history['val_mse']
# best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
# print('\nBest epoch: %d\n' % (best_epoch,))

# %%

model = tuner.hypermodel.build(best_hps)

# Retrain the model
history=model.fit(X_train, y_train, epochs=350, validation_data=(X_test, y_test), shuffle=True)

plot_history(history)

# model.save('models/ann1')
# joblib.dump([x_scaler, y_scaler], 'models/ann1/scalers.pkl')

#results = model.evaluate(X_test, y_test)

#print("test loss, test mae, test mse:", results)

# model = KerasRegressor(build_fn = lambda: create_model(best_hps))

# # Apply previous scaling to test dataset
# X_cv, y_cv, _, _ = standardize_data(X_shuf, y_shuf, x_scaler, y_scaler)

# train_sizes, train_scores, test_scores = learning_curve(model, X_cv, y_cv, cv=5, n_jobs=-1, verbose=3, scoring='neg_mean_squared_error', shuffle=True, random_state=SEED)
# plot_learning_curve(train_sizes, train_scores, test_scores)
# %%
from sklearn.utils import shuffle
from re import S
from tensorflow.keras import callbacks
from tensorflow.python.keras.backend import dtype
from tensorflow.python.keras.callbacks import History
from tensorflow.python.keras.engine.training_utils import batch_shuffle, call_metric_function

from tensorflow.python.keras.layers.core import Dropout
from tensorflow.python.ops.gen_array_ops import const, shape
from tensorflow.python.ops.gen_dataset_ops import parallel_interleave_dataset_v2
from functions import custom_loss, data_sampling, load_dataframes, select_features, select_features_multi, standardize_data, plot_history, standardize_force
from tensorflow import keras
from sklearn.model_selection import train_test_split, GroupShuffleSplit
import tensorflow as tf
import kerastuner as kt
import matplotlib.pyplot as plt
import pandas as pd
import constants
import joblib 
import random
import numpy as np
import keras.backend as kb

def create_model(hp):
    
    # Initializing sequential model
    model = keras.Sequential()

    # Defining input layer    
    model.add(keras.layers.Input(shape=(6,)))

    # Tuning the number of hidden layers and hidden units
    for i in range(hp.Int('num_layers', 1, 5)):

        hp_units = hp.Int('units_' + str(i), min_value=6, max_value=60, step=2)
        hp_activation = hp.Choice('activ_' + str(i), values=['relu','tanh','selu','elu'])
        hp_weight_init = hp.Choice('init_' + str(i), values=['truncated_normal', 'variance_scaling', 'orthogonal', 'glorot_normal', 'glorot_uniform'])
        #hp_l2 = hp.Choice('l2_reg_' + str(i),values=[1e-1, 1e-2, 1e-3, 1e-4])
        
        model.add(keras.layers.Dense(units=hp_units,
                                    activation= hp_activation,
                                    use_bias=True,
                                    bias_initializer='ones',
                                    kernel_initializer=hp_weight_init,
                                    #kernel_regularizer=keras.regularizers.L2(l2=0.001)
                                    )
        )
    
    # Defining the output layer
    model.add(keras.layers.Dense(units=3))

    # Tune the learning rate for the optimizer
    # Choose an optimal value from 0.01, 0.001, or 0.0001
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 0.0075, 0.005, 0.0025, 1e-3, 1e-4])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01),
                  loss=custom_loss,
                  run_eagerly=True
                  #metrics=['mae', 'mse']
                  )

    return model

# Loading data
df_list, _ = load_dataframes(constants.TRAIN_MULTI_DIR)

# Sampling data pass random seed for random sampling
sampled_dfs = data_sampling(df_list, constants.DATA_SAMPLES)

# plt.rcParams.update(constants.PARAMS)
# for i, df in enumerate(sampled_dfs):
#     plt.xlabel(r'$\varepsilon$')
#     plt.ylabel(r'$\sigma$ [MPa]')
#     plt.plot(df['exx_t+dt'], df['sxx_t+dt'], 'go--', linewidth=0.15, markersize=0.75)
    
# plt.show()

# Merging training data
data = pd.concat(sampled_dfs, axis=0, ignore_index=True)

# Reorganizing dataset by time increment
data = pd.concat([pd.concat([df.iloc[[index]] for df in sampled_dfs], axis=0) for index in sampled_dfs[0].index.tolist()], axis=0)

data_groups = [df for _, df in data.groupby('t')]
random.shuffle(data_groups)

data = pd.concat(data_groups).reset_index(drop=True)

X, y = select_features_multi(data)

# Shuffling dataset
#X_shuf, y_shuf = shuffle(X, y, random_state=constants.SEED)

# Splitting data into train and test datasets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=constants.TEST_SIZE, random_state=constants.SEED)

train_inds, test_inds = next(GroupShuffleSplit(test_size=constants.TEST_SIZE, n_splits=2, random_state = constants.SEED).split(data, groups=data['t']))

X_train = X.iloc[train_inds]
X_test = X.iloc[test_inds]

y_train = y.iloc[train_inds]
y_test = y.iloc[test_inds]

global_force_train = y_train[['fxx_t', 'fyy_t', 'fxy_t']]
global_force_test = y_test[['fxx_t', 'fyy_t', 'fxy_t']]

y_train = y_train.drop(['fxx_t', 'fyy_t', 'fxy_t'], axis=1)
y_test = y_test.drop(['fxx_t', 'fyy_t', 'fxy_t'], axis=1)

# Normalizing/Standardizing training dataset
X_train, y_train, x_scaler, y_scaler = standardize_data(X_train, y_train)
global_force_train, _ = standardize_force(global_force_train, y_scaler)

# Apply previous scaling to test dataset
X_test, y_test, _, _ = standardize_data(X_test, y_test, x_scaler, y_scaler)
global_force_test, _ = standardize_force(global_force_test, y_scaler)

#%%
# Defining tuner
tuner = kt.Hyperband(create_model,
                     objective='loss',
                     max_epochs=200,
                     factor=3,
                     executions_per_trial=2,
                     seed=constants.SEED,
                     directory='hyperband',
                     project_name='hyperparameter_tuning')

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=1, restore_best_weights=True, min_delta=1e-4)

# Performing hyperparameter search
tuner.search(X_train, y_train, epochs=200, validation_data=(X_test, y_test), shuffle=True, callbacks=[stop_early], batch_size=32)

# Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

# print(f"""\nThe hyperparameter search is complete:\n
# - Optimal number of hidden layers: {best_hps.get('num_layers')}
# """)

# for i in range(best_hps.get('num_layers')):
#     print('\tNeurons in hidden layer ' + str(i+1) + ': ' + str(best_hps.get('units_' + str(i))))
#     print('\t\t- Activation: ' + str(best_hps.get('activ_' + str(i))))
#     #print('\t\t- L2 regularization: ' + str(best_hps.get('l2_reg_' + str(i))))

# print(f"""
# - Optimal learning rate for the optimizer: {best_hps.get('learning_rate')}
# """)

#Build the model with the optimal hyperparameters and train it on the data
# model = tuner.hypermodel.build(best_hps)
# print(model.summary())
# history = model.fit(X_train, y_train, epochs=200, validation_data=(X_test, y_test), shuffle=True)

# val_acc_per_epoch = history.history['val_mse']
# best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
# print('\nBest epoch: %d\n' % (best_epoch,))

# %%

model = tuner.hypermodel.build(best_hps)

model.summary()

#  %%
## Custom training loop

# Training settings
optimizer = tf.keras.optimizers.Adam(kb.eval(model.optimizer.lr))
batch_size = 9
epochs = 400

# Preparing training and test datasets
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)

# Splitting global_force array according to batch_size
batch_indices_train = np.arange(batch_size, round(constants.DATA_SAMPLES*batch_size*(1-constants.TEST_SIZE)), batch_size)
batch_indices_test = np.arange(batch_size, round(constants.DATA_SAMPLES*batch_size*constants.TEST_SIZE), batch_size)

global_force_batches_train = np.array_split(global_force_train, batch_indices_train)
global_force_batches_test = np.array_split(global_force_test, batch_indices_test)

### tf.functions for training
@tf.function
def train_on_batch(X, y):
    # Open a GradientTape to record the operations run
    # during the forward pass, which enables auto-differentiation.
    with tf.GradientTape() as tape:
        # Run the forward pass of the layer.
        # The operations that the layer applies
        # to its inputs are going to be recorded
        # on the GradientTape.
        ŷ = model(X, training=True)
        # Compute the loss value for this minibatch.
        loss_value = custom_loss(tf.cast(y, dtype='float32'), ŷ)
    # Use the gradient tape to automatically retrieve
    # the gradients of the trainable variables with respect to the loss.
    grads = tape.gradient(loss_value, model.trainable_weights)
    # Run one step of gradient descent by updating
    # the value of the variables to minimize the loss.
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    return loss_value

@tf.function
def validate_on_batch(X, y):
    ŷ = model(X, training=False)
    loss_value = custom_loss(tf.cast(y, dtype='float32'), ŷ)
    return loss_value

# Model Checkpointing and better prints
best_loss = 99999

loss = np.zeros((epochs,1))
val_loss = np.zeros((epochs,1))
epochs_ = np.arange(0,epochs).reshape(epochs,1)

for epoch in range(0, epochs):
     # Iterate over the batches of the dataset.
    
    batch_loss = []
    for batch, (X, y) in enumerate(train_dataset):
        batch_loss.append(train_on_batch(X, tf.concat([y, global_force_batches_train[batch]], axis=1)))
        print('\rEpoch [%d/%d] Batch: %d%s' % (epoch + 1, epochs, batch, '.' * (batch % 10)), end='')
        
    loss[epoch,:] = np.mean(batch_loss)
    print('. loss: ' + str(np.mean(batch_loss)), end='')

    batch_val_loss = []    
    for batch_test, (X_test, y_test) in enumerate(test_dataset):
        batch_val_loss.append(validate_on_batch(X_test, tf.concat([y_test, global_force_batches_test[batch_test]], axis=1)))
    # val_loss = np.mean([np.mean(validate_on_batch(X_test, tf.concat([y_test, global_force_batches_test[batch_test]], axis=1)), axis=0) for batch_test, (X_test, y_test) in enumerate(test_dataset)])
    val_loss[epoch, :] = np.mean(batch_val_loss)
    print('. val_loss: ' + str(np.mean(batch_val_loss)))
    # if val_loss < best_loss:
    #     #model.save_weights('model.h5')
    #     best_loss = val_loss

# Retrain the model
#history=model.fit(X_train, y_train, epochs=200, validation_data=(X_test, y_test), shuffle='batch', batch_size=9)
history = pd.DataFrame(np.concatenate([epochs_, loss, val_loss], axis=1), columns=['epoch','loss','val_loss'])

plot_history(history, True)

model.save('models/ann3')
joblib.dump([x_scaler, y_scaler], 'models/ann3/scalers.pkl')

#results = model.evaluate(X_test, y_test)

#print("test loss, test mae, test mse:", results)

# model = KerasRegressor(build_fn = lambda: create_model(best_hps))

# # Apply previous scaling to test dataset
# X_cv, y_cv, _, _ = standardize_data(X_shuf, y_shuf, x_scaler, y_scaler)

# train_sizes, train_scores, test_scores = learning_curve(model, X_cv, y_cv, cv=5, n_jobs=-1, verbose=3, scoring='neg_mean_squared_error', shuffle=True, random_state=SEED)
# plot_learning_curve(train_sizes, train_scores, test_scores)
# %%

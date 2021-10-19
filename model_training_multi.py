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

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, deformation, stress, force, list_IDs, scaler_x=None, scaler_y=None, batch_size=9, shuffle=True):
        super().__init__()
        self.X = deformation.iloc[list_IDs]
        self.y = stress.iloc[list_IDs]
        self.f = force.iloc[list_IDs]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.list_IDs = list_IDs
        self.standardize()
        self.f_max = np.max(f.values, axis=0)
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
    
        indexes = np.array([self.indexes[index]+i for i in range(batch_size)])

        # Generate data
        X, y, f = self.__data_generation(np.random.permutation(indexes))

        return X, y, f

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(0, len(self.list_IDs), self.batch_size)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def standardize(self):
        idx = self.X.index
        self.X, self.y, self.scaler_x, self.scaler_y = standardize_data(self.X, self.y)
        self.f, self.scaler_f = standardize_force(self.f)

        self.X = pd.DataFrame(self.X, index=idx)
        self.y = pd.DataFrame(self.y, index=idx)
        self.f = pd.DataFrame(self.f, index=idx)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        X = np.asarray(self.X.iloc[list_IDs_temp], dtype=np.float32)
        y = np.asarray(self.y.iloc[list_IDs_temp], dtype=np.float32)
        f = np.asarray(self.f.iloc[list_IDs_temp], dtype=np.float32)
        
        return X, y, f

def create_model(hp):
    
    # Initializing sequential model
    model = keras.Sequential()

    # Defining input layer    
    model.add(keras.layers.Input(shape=(6,)))

    # Tuning the number of hidden layers and hidden units hp.Int('num_layers', 1, 5)
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

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss=custom_loss
                  #run_eagerly=True
                  #metrics=['mae', 'mse']
                  )

    return model

# Specifying random seed
random.seed(constants.SEED)

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

# Reorganizing dataset by time increment and applying first shuffling
data_groups = [df for _, df in data.groupby(['t'])]
random.shuffle(data_groups)

data = pd.concat(data_groups).reset_index(drop=True)

X, y, f = select_features_multi(data)

# Shuffling dataset
#X_shuf, y_shuf = shuffle(X, y, random_state=constants.SEED)

# Splitting data into train and test datasets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=constants.TEST_SIZE, random_state=constants.SEED)

partition = {"train": None, "test": None}

partition['train'], partition['test'] = next(GroupShuffleSplit(test_size=constants.TEST_SIZE, n_splits=2, random_state = constants.SEED).split(data, groups=data['t']))

batch_size = 9

train_generator = DataGenerator(X, y, f, partition["train"], batch_size, shuffle = True)
#test_generator = DataGenerator(X, y, f, partition['test'], train_generator.scaler_x, train_generator.scaler_y, batch_size, shuffle = True)

# X_train = X.iloc[train_inds]
# X_test = X.iloc[test_inds]

# y_train = y.iloc[train_inds]
# y_test = y.iloc[test_inds]

# global_force_train = y_train[['fxx_t', 'fyy_t', 'fxy_t']]
# global_force_test = y_test[['fxx_t', 'fyy_t', 'fxy_t']]

# y_train = y_train.drop(['fxx_t', 'fyy_t', 'fxy_t'], axis=1)
# y_test = y_test.drop(['fxx_t', 'fyy_t', 'fxy_t'], axis=1)

# # Normalizing/Standardizing training dataset
# X_train, y_train, x_scaler, y_scaler = standardize_data(X_train, y_train)
# global_force_train, _ = standardize_force(global_force_train, y_scaler)

# # Apply previous scaling to test dataset
# X_test, y_test, _, _ = standardize_data(X_test, y_test, x_scaler, y_scaler)
# global_force_test, _ = standardize_force(global_force_test, y_scaler)

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
#tuner.search(X_train, y_train, epochs=200, validation_data=(X_test, y_test), shuffle=True, callbacks=[stop_early], batch_size=32)

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
# Custom training loop

# Training settings
optimizer = tf.keras.optimizers.Adam(kb.eval(model.optimizer.lr))

n_batches_train = len(train_generator)
#n_batches_val = len(test_generator)
epochs = 50

# # Preparing training and test datasets
# train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)
# test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)

# # Splitting global_force array according to batch_size
# batch_indices_train = np.arange(batch_size, round(constants.DATA_SAMPLES*batch_size*(1-constants.TEST_SIZE)), batch_size)
# batch_indices_test = np.arange(batch_size, round(constants.DATA_SAMPLES*batch_size*constants.TEST_SIZE), batch_size)

# global_force_batches_train = np.array_split(global_force_train, batch_indices_train)
# global_force_batches_test = np.array_split(global_force_test, batch_indices_test)

## tf.functions for training
def train_on_batch(X, f, f_scale):
    # Open a GradientTape to record the operations run
    # during the forward pass, which enables auto-differentiation.
    with tf.GradientTape() as tape:
        # Run the forward pass of the layer.
        # The operations that the layer applies
        # to its inputs are going to be recorded
        # on the GradientTape.
        y_pred = model(X, training=True)
        # Compute the loss value for this minibatch.
        loss_value, internal_forces_stack, global_f = custom_loss(f, y_pred, f_scale, X[:,3:])
    # Use the gradient tape to automatically retrieve
    # the gradients of the trainable variables with respect to the loss.
    grads = tape.gradient(loss_value, model.trainable_weights)
    # Run one step of gradient descent by updating
    # the value of the variables to minimize the loss.
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    return loss_value, y_pred, internal_forces_stack, global_f

def validate_on_batch(X, f):
    y_pred = model(X, training=False)
    loss_value = custom_loss(f, y_pred)
    return loss_value

train_loss = np.zeros(shape=(epochs,1), dtype=np.float32)
val_loss = np.zeros(shape=(epochs,1), dtype=np.float32)
epochs_ = np.arange(0,epochs).reshape(epochs,1)

logs = []

def train(train_dataset, n_epochs, train_on_batch):

    train_loss = np.zeros(shape=(epochs,1), dtype=np.float32)
    val_loss = np.zeros(shape=(epochs,1), dtype=np.float32)
    epochs_ = np.arange(0,epochs).reshape(epochs,1) 

    for epoch in range(n_epochs):
        # Iterate over the batches of the dataset.
        train_dataset.on_epoch_end()
        train_dataset.on_epoch_end()

        #epoch_loss_avg = tf.keras.metrics.Mean() # Keeping track of the training loss
        #epoch_val_loss_avg = tf.keras.metrics.Mean()
        all_batches = []
        for batch in range(n_batches_train):
            X_train, y_train, f_train = train_dataset[batch]
            X_train = tf.convert_to_tensor(X_train)
            f_train = tf.convert_to_tensor(f_train)
            y_train = tf.convert_to_tensor(y_train)
            loss_value, y_pred, internal_forces_stack, global_f = train_on_batch(X_train, f_train, train_dataset.f_max)
            all_batches.append(tf.reduce_mean(loss_value))
            #all_batches.append(tf.reduce_mean(train_on_batch(X_train, f_train)))
            #loss_batch = np.mean(train_on_batch(X_train, tf.concat([y_train, f_train], axis=1)))
            print('\rEpoch [%d/%d] Batch: %d' % (epoch + 1, epochs, batch), end='')
            #epoch_loss_avg(loss_batch)
            epoch_arr = np.array([epoch] * batch_size).reshape(batch_size,1)
            batch_arr = np.array([batch] * batch_size).reshape(batch_size,1)
            vars = [epoch_arr, batch_arr, kb.eval(y_train), kb.eval(y_pred), kb.eval(f_train), kb.eval(internal_forces_stack)]
            logs.append(np.concatenate(vars, axis=1))
            # log_dict[epoch][batch]['y_pred'] = tf.make_ndarray(y_pred)

        #train_loss[epoch]=epoch_loss_avg.result()
        train_loss[epoch]=np.mean(all_batches)
        print('. loss: ' + str(train_loss[epoch][0]))

        # all_batches_val = []
        # for batch_test in range(n_batches_val):
        #     X_test, y_test, f_test = test_generator[batch_test]
        #     X_test = tf.convert_to_tensor(X_test)
        #     f_test = tf.convert_to_tensor(f_test)
        #     y_test = tf.convert_to_tensor(y_test)
        #     all_batches_val.append(tf.reduce_mean(validate_on_batch(X_test, f_test)))
        #     #epoch_val_loss_avg(val_loss_batch)

        # val_loss[epoch] = np.mean(all_batches_val)
        # #val_loss[epoch] = epoch_val_loss_avg.result()
        # print('. val_loss: ' + str(val_loss[epoch][0]))

    return epochs_, train_loss, val_loss

tf_train_on_batch = tf.function(train_on_batch)
epochs_, train_loss, val_loss = train(train_generator, epochs, tf_train_on_batch)

file_name = 'logs.csv'

np.savetxt(file_name, 
        np.array(logs).reshape(-1,logs[0].shape[-1]),
        fmt=['%i', '%i', '%.3f', '%.3f', '%.3f', '%.3f', '%.3f', '%.3f', '%.3f', '%.3f', '%.3f', '%.3f'], 
        delimiter='\t',
        comments='',
        header='e\tb\ty_t_1\ty_t_2\ty_t_3\ty_p_1\ty_p_2\ty_p_3\tf_t_1\tf_t_2\tif_1\tif_2')

# # Retrain the model
# #history=model.fit(X_train, y_train, epochs=200, validation_data=(X_test, y_test), shuffle='batch', batch_size=9)

history = pd.DataFrame(np.concatenate([epochs_, train_loss, val_loss], axis=1), columns=['epoch','loss','val_loss'])

plot_history(history, True)

model.save('models/ann3')
#joblib.dump([train_generator.scaler_x, train_generator.scaler_y, train_generator.scaler_f], 'models/ann3/scalers.pkl')

#results = model.evaluate(X_test, y_test)

#print("test loss, test mae, test mse:", results)

# model = KerasRegressor(build_fn = lambda: create_model(best_hps))

# # Apply previous scaling to test dataset
# X_cv, y_cv, _, _ = standardize_data(X_shuf, y_shuf, x_scaler, y_scaler)

# train_sizes, train_scores, test_scores = learning_curve(model, X_cv, y_cv, cv=5, n_jobs=-1, verbose=3, scoring='neg_mean_squared_error', shuffle=True, random_state=SEED)
# plot_learning_curve(train_sizes, train_scores, test_scores)
# %%

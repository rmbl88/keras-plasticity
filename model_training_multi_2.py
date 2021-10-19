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
from functions import custom_loss, data_sampling, load_dataframes, select_features, select_features_multi, standardize_data, plot_history, standardize_
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
from autograd_minimize.tf_wrapper import tf_function_factory
from autograd_minimize import minimize
from keras.wrappers.scikit_learn import KerasRegressor


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, deformation, stress, force, coord, list_IDs, batch_size, shuffle):
        super().__init__()
        self.X = deformation.iloc[list_IDs]
        self.y = stress.iloc[list_IDs]
        self.f = force.iloc[list_IDs]
        self.coord = coord[['x','y']].iloc[list_IDs]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.list_IDs = list_IDs
        #self.standardize()
        self.f_max = np.max(f.values, axis=0)
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
    
        indexes = np.array([self.indexes[index]+i for i in range(self.batch_size)])

        # Generate data
        X, y, f, coord = self.__data_generation(np.random.permutation(indexes))

        return X, y, f, coord

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(0, len(self.list_IDs), self.batch_size)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def standardize(self):
        idx = self.X.index
        self.X, self.y, self.scaler_x, self.scaler_y = standardize_data(self.X, self.y)
        self.f, self.scaler_f = standardize_(self.f)
        #self.coord, self.scaler_coord = standardize_(self.coord[['x','y']])

        self.X = pd.DataFrame(self.X, index=idx)
        self.y = pd.DataFrame(self.y, index=idx)
        self.f = pd.DataFrame(self.f, index=idx)
        #self.coord = pd.DataFrame(self.coord, index=idx)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        X = np.asarray(self.X.iloc[list_IDs_temp], dtype=np.float32)
        y = np.asarray(self.y.iloc[list_IDs_temp], dtype=np.float32)
        f = np.asarray(self.f.iloc[list_IDs_temp], dtype=np.float32)
        coord = np.asarray(self.coord.iloc[list_IDs_temp], dtype=np.float32)
        return X, y, f, coord

def create_model():
    
    # Initializing sequential model
    model = keras.Sequential()

    # Defining input layer    
    model.add(keras.layers.Input(shape=(6,)))

    model.add(keras.layers.Dense(units=10,
                                    activation= 'relu',
                                    use_bias=True,
                                    bias_initializer='ones',
                                    kernel_initializer=tf.keras.initializers.random_uniform(minval=-0.12, maxval=0.12)
                                    )
        )
    
    # Defining the output layer
    model.add(keras.layers.Dense(units=3))

    # Tune the learning rate for the optimizer
    # Choose an optimal value from 0.01, 0.001, or 0.0001
    #hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 0.0075, 0.005, 0.0025, 1e-3, 1e-4])
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.1),
                loss=custom_loss,
                run_eagerly=True,
                #metrics=['mae', 'mse']
                )

    return model

# Specifying random seed
random.seed(constants.SEED)

# Loading data
df_list, _ = load_dataframes(constants.TRAIN_MULTI_DIR)

# Sampling data pass random seed for random sampling
sampled_dfs = data_sampling(df_list, constants.DATA_SAMPLES)

# Merging training data
data = pd.concat(sampled_dfs, axis=0, ignore_index=True)

# Reorganizing dataset by time increment and applying first shuffling
data_groups = [df for _, df in data.groupby(['tag'])]
#random.shuffle(data_groups)

data = pd.concat(data_groups).reset_index(drop=True)

X, y, f, coord = select_features_multi(data)

#partition = {"train": None, "test": None}

#partition['train'], partition['test'] = next(GroupShuffleSplit(test_size=constants.TEST_SIZE, n_splits=2, random_state = constants.SEED).split(data, groups=data['t']))

batch_size = len(data_groups[0])

#batch_size = 9

train_generator = DataGenerator(X, y, f, coord, X.index.tolist(), batch_size, False)
# train_generator = DataGenerator(X, y, f, coord, partition["train"], batch_size, False)
#test_generator = DataGenerator(X, y, f, partition['test'], train_generator.scaler_x, train_generator.scaler_y, batch_size, shuffle = True)

#model = create_model()

#model.summary()

#  %%
# Custom training loop

# Minimization

learning_rate = kb.eval(model.optimizer.lr)

epochs = 1000

lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.1,
    decay_steps=10000,
    decay_rate=0.9)
# Training settings
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
# optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr_schedule)
n_batches_train = len(train_generator)
#n_batches_val = len(test_generator)

## tf.functions for training
def train_on_batch(X, y_true):
    # Open a GradientTape to record the operations run
    # during the forward pass, which enables auto-differentiation.
   
    with tf.GradientTape() as tape:
        
        # Run the forward pass of the layer.
        # The operations that the layer applies
        # to its inputs are going to be recorded
        # on the GradientTape.
        y_pred = model(X, training=True)
        
        # Compute the loss value for this minibatch.
        loss_value = custom_loss(y_true, y_pred)
        
    # Use the gradient tape to automatically retrieve
    # the gradients of the trainable variables with respect to the loss.
    grads = tape.gradient(loss_value, model.trainable_weights)
    # Run one step of gradient descent by updating
    # the value of the variables to minimize the loss.
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    return loss_value

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

        #epoch_loss_avg = tf.keras.metrics.Mean() # Keeping track of the training loss
        #epoch_val_loss_avg = tf.keras.metrics.Mean()
        all_batches = []
        for batch in range(n_batches_train):
            X_train, y_train, f_train, coord = train_dataset[batch]
            X_train = tf.convert_to_tensor(X_train, dtype='float32')
            f_train = tf.convert_to_tensor(f_train, dtype='float32')
            y_train = tf.convert_to_tensor(y_train, dtype='float32')
            coord = tf.convert_to_tensor(coord, dtype='float32')
            loss_value = train_on_batch(X_train, tf.concat([y_train,f_train,coord], axis=1))
            all_batches.append(tf.reduce_mean(loss_value))
            #all_batches.append(tf.reduce_mean(train_on_batch(X_train, f_train)))
            #loss_batch = np.mean(train_on_batch(X_train, tf.concat([y_train, f_train], axis=1)))
            print('\rEpoch [%d/%d] Batch: %d' % (epoch + 1, epochs, batch+1), end='')
            #epoch_loss_avg(loss_batch)
            # epoch_arr = np.array([epoch] * batch_size).reshape(batch_size,1)
            # batch_arr = np.array([batch] * batch_size).reshape(batch_size,1)
            # s_true = train_generator.scaler_y.inverse_transform(kb.eval(y_train))
            # s_pred = train_generator.scaler_y.inverse_transform(kb.eval(y_pred))
            # vars = [epoch_arr, batch_arr, s_true, s_pred, kb.eval(f_train), kb.eval(global_f_pred)*np.ones((batch_size,3))]
            # logs.append(np.concatenate(vars, axis=1))
            #log_dict[epoch][batch]['y_pred'] = tf.make_ndarray(y_pred)

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

if constants.PRE_TRAINING:
    model.fit_generator(train_generator, epochs=epochs,verbose=1)
else:
    #tf_train_on_batch = tf.function(train_on_batch)
    epochs_, train_loss, val_loss = train(train_generator, epochs, train_on_batch)

    # file_name = 'logs.csv'

    # np.savetxt(file_name, 
    #         np.array(logs).reshape(-1,logs[0].shape[-1]),
    #         fmt=['%i', '%i', '%.3f', '%.3f', '%.3f', '%.3f', '%.3f', '%.3f', '%.3f', '%.3f', '%.3f', '%.3f', '%.3f', '%.3f'], 
    #         delimiter='\t',
    #         comments='',
    #         header='epoch\tbatch\ts_11\ts_22\ts_12\ts_p_11\ts_p_22\ts_p_12\tfg_1\tfg_2\tfg_12\tfg_p_1\tfg_p_2\tfg_p_12')

    history = pd.DataFrame(np.concatenate([epochs_, train_loss, val_loss], axis=1), columns=['epoch','loss','val_loss'])

    plot_history(history, True)

if constants.PRE_TRAINING:
    model.save_weights('models/ann3/weights.h5')
    model.save('models/ann3')
    joblib.dump([train_generator.scaler_x, train_generator.scaler_y], 'models/ann3/scalers.pkl')
else:
    model.save('models/ann3')
    #joblib.dump([train_generator.scaler_x, train_generator.scaler_y], 'models/ann3/scalers.pkl')



#results = model.evaluate(X_test, y_test)

#print("test loss, test mae, test mse:", results)

# model = KerasRegressor(build_fn = lambda: create_model(best_hps))

# # Apply previous scaling to test dataset
# X_cv, y_cv, _, _ = standardize_data(X_shuf, y_shuf, x_scaler, y_scaler)

# train_sizes, train_scores, test_scores = learning_curve(model, X_cv, y_cv, cv=5, n_jobs=-1, verbose=3, scoring='neg_mean_squared_error', shuffle=True, random_state=SEED)
# plot_learning_curve(train_sizes, train_scores, test_scores)
# %%

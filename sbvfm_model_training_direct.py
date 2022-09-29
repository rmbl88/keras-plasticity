# ---------------------------------
#    Library and function imports
# ---------------------------------
from audioop import mul
import cProfile, pstats
from doctest import master
import os
from pyexpat import model
import shutil
import joblib
from sklearn import model_selection
from constants import *
from functions import (
    InputConvexNN,
    layer_wise_lr,
    sbvf_loss,
    load_dataframes,
    prescribe_u,
    select_features_multi,
    standardize_data,
    plot_history,
    read_mesh,
    global_strain_disp,
    param_deltas,
    global_dof)
from functions import (
    weightConstraint,
    EarlyStopping,
    NeuralNetwork,
    Element)
from functools import partial
import copy
from torch import nn
import tensorflow as tf
import pandas as pd
import random
import numpy as np
import math
import torch
import time
import itertools
import wandb
from torch.autograd.functional import jacobian
import time

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.ticker import MaxNLocator
import multiprocessing
import time
import random
from tkinter import *
import geotorch
import matplotlib.pyplot as plt

# -----------------------------------------
#   DEPRECATED IMPORTS
# -----------------------------------------
# from torch.nn.utils import (
#   parameters_to_vector as Params2Vec,
#   vector_to_parameters as Vec2Params
# )

# ----------------------------------------
#        Class definitions
# ----------------------------------------
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, deformation, stress, force, coord, info, list_IDs, batch_size, shuffle=False, std=True, t_pts=1, scaler=None):
        super().__init__()

        self.X = deformation.iloc[list_IDs].reset_index(drop=True)
        self.y = stress.iloc[list_IDs].reset_index(drop=True)
        self.f = force.iloc[list_IDs].reset_index(drop=True)
        self.coord = coord[['dir','id','cent_x','cent_y','area']].iloc[list_IDs].reset_index(drop=True)
        self.tag = info['tag'].iloc[list_IDs].reset_index(drop=True)
        self.t = info[['inc','t','d_exx','d_eyy','d_exy']].iloc[list_IDs].reset_index(drop=True)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.list_IDs = list_IDs
        self.std = std
        self.t_pts = t_pts
        self.scaler_x = scaler

        if self.std == True:
            self.standardize()

        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = np.array([self.indexes[index]+i for i in range(self.batch_size)])  # Generate indexes of the batch

        # Generate data according to batch size specifications
        if self.shuffle == True:
            index_groups = np.array_split(indexes, self.t_pts)
            #permuted = [np.random.permutation(index_groups[i]) for i in range(len(index_groups))]
            permuted = np.random.permutation(index_groups)
            indexes = np.hstack(permuted)

        x, y, f, coord, tag, t = self.__data_generation(indexes)

        return x, y, f, coord, tag, t

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(0, len(self.list_IDs), self.batch_size)
        #if self.shuffle == True:
        np.random.shuffle(self.indexes)

    def standardize(self):
        'Standardizes neural network input data'
        idx = self.X.index
        self.X, _, _, self.scaler_x, _, _ = standardize_data(self.X, self.y, self.f)

        self.X = pd.DataFrame(self.X, index=idx)
        #self.y = pd.DataFrame(self.y, index=idx)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        x = np.asarray(self.X.iloc[list_IDs_temp], dtype=np.float64)
        y = np.asarray(self.y.iloc[list_IDs_temp], dtype=np.float64)
        f = np.asarray(self.f.iloc[list_IDs_temp], dtype=np.float64)
        coord = np.asarray(self.coord.iloc[list_IDs_temp], dtype=np.float64)
        tag = self.tag.iloc[list_IDs_temp]
        t = self.t.iloc[list_IDs_temp]

        return x, y, f, coord, tag, t

# -------------------------------
#       Plotting classes
# -------------------------------
def plot():    #Function to create the base plot, make sure to make global the lines, axes, canvas and any part that you would want to update later

    global v_loss,t_loss,ax_1,ax_2,canvas,ax1_background,ax1_background,ax2_background,train_data,val_data,epochs

    fig = matplotlib.figure.Figure(figsize=(15, 6), dpi=100, tight_layout=True)
    
    ax_1 = fig.add_subplot(121)
    ax_2 = fig.add_subplot(122)

    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()
    canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)
    canvas._tkcanvas.pack(side=TOP, fill=BOTH, expand=1)
      

    v_loss, = ax_1.plot([],label='test_loss')
    t_loss, = ax_1.plot([],label='train_loss')
    ax_1.set_title('Learning curves')
    
    l0_loss, = ax_2.plot([],label='l0_loss')
    l1_loss, = ax_2.plot([],label='l1_loss')
    lw_loss, = ax_2.plot([],label='lw_loss')
    ax_2.set_title('Constraints loss')
        
    for ax in [ax_1,ax_2]:
        ax.set_yscale('log')
        ax.set_xlim([0, 2])
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(loc='best')
    
  
    ax1_background = canvas.copy_from_bbox(ax_1.get_figure().bbox)
    ax2_background = canvas.copy_from_bbox(ax_2.get_figure().bbox)

    train_data = [-1]
    val_data = [-1]
    epochs = [-1]

def updateplot(q):

    if q.empty():       #Try to check if there is data in the queue
        
        pass
    
    else:
        
        result, tag = q.get_nowait()
        
        if result[0] != 'Q':
            if tag == 'loss':

                #canvas.restore_region(ax2_background)
                epochs.append(epochs[-1]+1)
                train_data.append(result[0])
                val_data.append(result[1])
                #d1 = np.append(t_loss.get_ydata(),result[0])
                #d2 = np.append(v_loss.get_ydata(),result[1])

                #epochs = list(range(len(d1)))
                t_loss.set_data(epochs,train_data)
                # t_loss.set_xdata(epochs)
                # t_loss.set_ydata(d1)
                v_loss.set_data(epochs,val_data)
                # v_loss.set_xdata(epochs)
                # v_loss.set_ydata(d2)

                lower = min([min(a) for a in [train_data,val_data]])
                upper = max([max(a) for a in [train_data,val_data]])

                ax_1.set_xlim([0, max(epochs)+5])
                ax_1.set_ylim([lower-0.5*abs(lower), upper+0.1*abs(upper)])

                
                #canvas.restore_region(ax1_background)
                ax_1.draw_artist(ax_1.patch)
                ax_1.draw_artist(t_loss)
                ax_1.draw_artist(v_loss)
                #canvas.blit(ax_1.clipbox)
                # canvas.blit(ax_2.clipbox)
                #canvas.update()
                #canvas.flush_events()

            else:
                a=0
                # w_int_=np.array(list(result['w_int'].values()))
                # w_ext_=np.array(list(result['w_ext'].values()))
                # w_int_real_=np.array(list(result['w_int_real'].values()))
                # t = np.array(list(result['w_int'].keys()))

                # lower = min([min(a) for a in [w_int_,w_ext_,w_int_real_]])
                # upper = max([max(a) for a in [w_int_,w_ext_,w_int_real_]])

                # ax.set_title(tag)
                # w_ext.set_xdata(t)
                # w_ext.set_ydata(w_ext_)
                # w_int.set_xdata(t)
                # w_int.set_ydata(w_int_)
                # w_int_real.set_xdata(t)
                # w_int_real.set_ydata(w_int_real_)

                # ax.set_xlim([min(t), max(t)+10])
                # ax.set_ylim([lower-0.2*abs(lower), upper+0.1*abs(upper)])

                # ax.draw_artist(ax.patch)
                # ax.draw_artist(w_ext)
                # ax.draw_artist(w_int)
                # ax.draw_artist(w_int_real)
                # canvas.draw()
                # canvas.flush_events()

            #canvas.draw()
            #canvas.flush_events()
            # window.after(1,updateplot,q)
    
    canvas.draw()
    canvas.flush_events()
    
    window.after(1,updateplot,q)


# -------------------------------
#       Method definitions
# -------------------------------

def batch_jacobian(f, x):
    f_sum = lambda x: torch.sum(f(x), axis=0)
    return jacobian(f_sum, x,create_graph=True).permute(1,0,2)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(q):

    def get_grad_norm(model):
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        return total_norm

    def init_weights(m):
        '''
        Performs the weight initialization of a neural network

        Parameters
        ----------
        m : NeuralNetwork object
            Neural network model, instance of NeuralNework class
        '''
        if isinstance(m, nn.Linear) and (m.bias != None):
            torch.nn.init.kaiming_normal_(m.weight) #RELU
            #torch.nn.init.xavier_normal(m.weight)
            #torch.nn.init.zeros_(m.bias)
            #torch.nn.init.ones_(m.bias)
            m.bias.data.fill_(0.01)

    def train_loop(dataloader, model, loss_fn, optimizer):
        '''
        Custom loop for neural network training, using mini-batches

        '''

        num_batches = len(dataloader)
        losses = torch.zeros(num_batches)
        
        g_norm = []

        l0_loss = torch.zeros_like(losses) 
        l1_loss = torch.zeros_like(losses)
        wp_loss = torch.zeros_like(losses)
        ch_loss = torch.zeros_like(losses)

        t_pts = dataloader.t_pts
        n_elems = batch_size // t_pts
               
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for batch in range(num_batches):

            # Extracting variables for training
            X_train, y_train, _, _, _, inc = dataloader[batch]

            # Converting to pytorch tensors
            X_train = torch.from_numpy(X_train)
            y_train = torch.from_numpy(y_train)
            d_e = torch.reshape(torch.from_numpy(inc[['d_exx','d_eyy','d_exy']].values),[t_pts,n_elems,3,1])

            #d_e = torch.reshape(torch.from_numpy(inc[['d_exx','d_eyy','d_exy']].values),[t_pts,n_elems,3,1])
            
            # Computing model stress prediction
            pred=model(X_train)

            l = torch.reshape(pred,[t_pts,n_elems,6])

            # Cholesky decomposition
            L = torch.zeros([t_pts, n_elems, 3, 3])
            tril_indices = torch.tril_indices(row=3, col=3, offset=0)
            L[:, :, tril_indices[0], tril_indices[1]] = l[:,:]
            # Tangent matrix
            H = L @torch.transpose(L,2,3)

            # Stress increment
            d_s = (H @ d_e).squeeze()
            s = torch.cumsum(d_s,0).reshape([-1,3])

            # s_ = copy.deepcopy(s.detach())
            # # Equivalent stress
            # mises = torch.sqrt(torch.square(s_[:,0]) - s_[:,0]*s_[:,1] + torch.square(s_[:,1]) + 3 * torch.square(s_[:,-1]))
            # # Yield transition
            # yield_transition = torch.sigmoid((torch.square(mises)-160**2))
            
            # yield_pt = torch.nonzero(yield_transition)

            # if yield_pt.shape[0] != 0:
            #     yield_pt = yield_pt[0]
            
            #     d_e_p = torch.zeros_like(s)
            #     d_e_p[yield_pt:,:] = d_e.squeeze().reshape([-1,3])[yield_pt:,:]
            
            #     # Plastic power
            #     w_p = torch.sum(s * d_e_p,-1)
            
            # else:
            #     w_p = torch.tensor(0.0)

            # l_wp = torch.mean(torch.nn.functional.relu(-w_p))

            
            # Stress at t=0
            s_0 = s[:9,:]
            l_0 = torch.mean(torch.square(s_0))
            
            l_1 = torch.mean(torch.nn.functional.relu(-(d_e.transpose(2,3) @ H @ d_e)))

            l_cholesky = torch.mean(nn.functional.relu(-torch.diagonal(L, offset=0, dim1=2, dim2=3)))

            # e_p = copy.deepcopy(e)
            # e_p[:,:,0:2] -= ((1/3)*torch.sum(e[:,:,0:2],-1,keepdim=True))
            # d_ep_dt = torch.cat([torch.zeros(1,n_elems,3),torch.diff(e_p,dim=0)/0.02])
            
            # w_ep = torch.sum(torch.reshape(pred,[t_pts,n_elems,3]) * d_ep_dt,-1)

            # l_w = torch.mean(torch.nn.functional.relu(-w_ep))
            m = torch.max(torch.abs(pred.detach()),dim=0).values
            m_= torch.max(torch.abs(y_train),dim=0).values
            # # Computing loss
            
            
            loss = (loss_fn(s[:,0], y_train[:,0]) + loss_fn(s[:,1], y_train[:,1]) + loss_fn(s[:,2], y_train[:,2])) + l_1 + 500*l_cholesky
           

            # Backpropagation and weight's update
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # # Gradient clipping - as in https://github.com/pseeth/autoclip
            # g_norm.append(get_grad_norm(model))
            # clip_value = np.percentile(g_norm, 25)
            #nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

            optimizer.step()
            scheduler.step()
            #model.apply(constraints)

            # Saving loss values
            losses[batch] = loss.detach().item()
            l0_loss[batch] = l_0.detach().item()
            l1_loss[batch] = l_1.detach().item()
            ch_loss[batch] = l_cholesky.detach().item()
            wp_loss[batch] = 0
            

            g_norm.append(get_grad_norm(model))
            #l += loss

            print('\r>Train: %d/%d' % (batch + 1, num_batches), end='')
        #l.backward()
        #optimizer.step()
        #scheduler.step()
                

        # get_sbvfs(copy.deepcopy(model), sbvf_generator)
        # return losses, v_work_real, v_disp, v_strain, v_u
        #-----------------------------
        return losses, np.mean(g_norm), l0_loss, l1_loss, ch_loss, wp_loss

    def test_loop(dataloader, model, loss_fn):

        #global w_virt

        num_batches = len(dataloader)
        test_losses = torch.zeros(num_batches)

        #n_vfs = v_strain.shape[0]
        t_pts = dataloader.t_pts
        n_elems = batch_size // t_pts
        model.eval()
        with torch.no_grad():

            for batch in range(num_batches):

                # Extracting variables for testing
                X_test, y_test, _, _, _, inc = dataloader[batch]

                # Converting to pytorch tensors
                if dataloader.scaler_x is not None:
                    X_test = torch.from_numpy(dataloader.scaler_x.transform(X_test))
                else:
                    X_test = torch.tensor(X_test, dtype=torch.float64)

                y_test = torch.from_numpy(y_test)

                d_e = torch.reshape(torch.from_numpy(inc[['d_exx','d_eyy','d_exy']].values),[t_pts,n_elems,3,1])

                pred = model(X_test)
                
                l = torch.reshape(pred,[t_pts,n_elems,6])

                L = torch.zeros([t_pts, n_elems, 3, 3])
                tril_indices = torch.tril_indices(row=3, col=3, offset=0)
                L[:, :, tril_indices[0], tril_indices[1]] = l[:,:]
                H = L @torch.transpose(L,2,3)

                d_s = (H @ d_e).squeeze()
                s = torch.cumsum(d_s,0).reshape([-1,3])

                # Computing losses
                test_loss = loss_fn(s[:,0], y_test[:,0]) + loss_fn(s[:,1], y_test[:,1]) + loss_fn(s[:,2], y_test[:,2])
                
                test_losses[batch] = test_loss
               

                print('\r>Test: %d/%d' % (batch + 1, num_batches), end='')

        #get_sbvfs(copy.deepcopy(model), dataloader, isTrain=False)
        return test_losses
#----------------------------------------------------

    # Default floating point precision for pytorch
    torch.set_default_dtype(torch.float64)

    # Specifying random seed
    os.environ['PYTHONHASHSEED']=str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Loading data
    df_list, _ = load_dataframes(TRAIN_MULTI_DIR)

    # Merging training data
    data = pd.concat(df_list, axis=0, ignore_index=True).dropna(axis=0)

    T_PTS = len(set(data['t']))

    # Performing test/train split
    partition = {"train": None, "test": None}

    # if T_PTS==DATA_SAMPLES:
    # Reorganizing dataset by tag, subsequent grouping by time increment
    data_by_tag = [df for _, df in data.groupby(['tag'])]
    random.shuffle(data_by_tag)
    data_by_t = [[df for _, df in group.groupby(['t'])] for group in data_by_tag]
    #random.shuffle(data_by_t)
    data_by_batches = list(itertools.chain(*data_by_t))
    #random.shuffle(data_by_batches)

    batch_size = len(data_by_batches[0]) * T_PTS

    data = pd.concat(data_by_batches).reset_index(drop=True)

    trials = list(set(data['tag'].values))
    random.shuffle(trials)
    test_trials = random.sample(trials, math.floor(len(trials)*TEST_SIZE))
    train_trials = list(set(trials).difference(test_trials))

    partition['train'] = data[data['tag'].isin(train_trials)].index.tolist()
    partition['test'] = data[data['tag'].isin(test_trials)].index.tolist()

    # Selecting model features
    X, y, f, coord, info = select_features_multi(data)

    # Preparing data generators for mini-batch training

    #partition['train'] = data[data['tag']=='m80_b80_x'].index.tolist()
    train_generator = DataGenerator(X, y, f, coord, info, partition["train"], batch_size, shuffle=False, std=True, t_pts=T_PTS)

    test_generator = DataGenerator(X, y, f, coord, info, partition['test'], batch_size, shuffle=False, std=False, t_pts=T_PTS, scaler=train_generator.scaler_x)

    # Model variables
    N_INPUTS = X.shape[1]
    N_OUTPUTS = y.shape[1] + 3

    N_UNITS = [12,12,12]
    H_LAYERS = len(N_UNITS)

    WANDB_LOG = True

    model_1 = NeuralNetwork(N_INPUTS, N_OUTPUTS, N_UNITS, H_LAYERS)

    model_1.apply(init_weights)

    # clip_value = 100
    # for p in model_1.parameters():
    #     p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))

    # Training variables
    epochs = 10000

    # Optimization variables
    # Optimization variables
    learning_rate = 0.002
    lr_mult = 1.0

    params = layer_wise_lr(model_1, lr_mult=lr_mult, learning_rate=learning_rate)

    loss_fn = torch.nn.MSELoss()

    
    weight_decay=0.002

    optimizer = torch.optim.AdamW(params=params, weight_decay=weight_decay)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=15, factor=0.65, min_lr=[params[i]['lr']*0.002 for i in range(len(params))])
    #[params[i]['lr']*0.025 for i in range(len(params))]
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,base_lr=0.002, max_lr=0.007, mode='exp_range',gamma=0.99994,cycle_momentum=False,step_size_up=len(train_generator)*20,step_size_down=len(train_generator)*200)
    #constraints=weightConstraint(cond='plastic')


    # Container variables for history purposes
    train_loss = []
    v_work = []
    val_loss = []
    epochs_ = []

    l0_loss = []
    l1_loss = []
    ch_loss = []
    wp_loss = []
    # j2_loss = []

    # Initializing the early_stopping object
    early_stopping = EarlyStopping(patience=500, path='temp/checkpoint.pt', verbose=True)

    if WANDB_LOG:
        config = {
            "inputs": N_INPUTS,
            "outputs": N_OUTPUTS,
            "hidden_layers": H_LAYERS,
            "hidden_units": "/".join(str(x) for x in N_UNITS),        
            "epochs": epochs,
            "lr": learning_rate,
            "l2_reg": weight_decay
        }
        run=wandb.init(project="direct_training", entity="rmbl",config=config)
        wandb.watch(model_1,log='all')

    for t in range(epochs):

        print('\r--------------------\nEpoch [%d/%d]' % (t + 1, epochs))

        epochs_.append(t+1)

        #Shuffling batches
        for generator in [train_generator,test_generator]:
            generator.on_epoch_end()

        # Train loop
        start_train = time.time()

        #--------------------------------------------------------------
        batch_losses, grad_norm, l_0, l_1, l_ch, l_wp = train_loop(train_generator, model_1, loss_fn, optimizer)
        #--------------------------------------------------------------

        train_loss.append(torch.mean(batch_losses))
        l0_loss.append(torch.mean(l_0))
        l1_loss.append(torch.mean(l_1))
        ch_loss.append(torch.mean(l_ch))
        wp_loss.append(torch.mean(l_wp))
        
        
        end_train = time.time()

        #Apply learning rate scheduling if defined
        try:
            #scheduler.step(train_loss[t])
            print('. t_loss: %.6e -> lr: %.4e | l0: %.4e | l1: %.4f | lw: %.4f -- %.3fs' % (train_loss[t], scheduler._last_lr[0], l0_loss[t], l1_loss[t], ch_loss[t], end_train - start_train))
        except:
            print('. t_loss: %.6e -- %.3fs' % (train_loss[t], v_work[t], end_train - start_train))

        # Test loop
        start_test = time.time()

        #-----------------------------------------------------------------------------------
        batch_val_losses = test_loop(test_generator, model_1, loss_fn)
        #-----------------------------------------------------------------------------------

        val_loss.append(torch.mean(batch_val_losses).item())

        end_test = time.time()

        print('. v_loss: %.6e -- %.3fs' % (val_loss[t], end_test - start_test))
        
        if t%9==0:
            q.put(([train_loss[t],val_loss[t]],'loss'))

        if t > 200:
            
            early_stopping(val_loss[t], model_1)

        if WANDB_LOG:
            wandb.log({
                'epoch': t,
                'l_rate': scheduler._last_lr[0],
                'train_loss': train_loss[t],
                'test_loss': val_loss[t],
                's(0)_error': l0_loss[t],
                'drucker_error': l1_loss[t],
                'cholesky_error': ch_loss[t],
                'plastic_power': wp_loss[t],
                #'j2_error': j2_loss[t]
            })


        if  early_stopping.early_stop:
            print("Early stopping")
            break
    

    q.put((['Q'],'empty'))
    print("Done!")

    # load the last checkpoint with the best model
    #model_1.load_state_dict(torch.load('temp/checkpoint.pt'))

    epochs_ = np.reshape(np.array(epochs_), (len(epochs_),1))
    train_loss = np.reshape(np.array(train_loss), (len(train_loss),1))
    val_loss = np.reshape(np.array(val_loss), (len(val_loss),1))

    history = pd.DataFrame(np.concatenate([epochs_, train_loss, val_loss], axis=1), columns=['epoch','loss','val_loss'])


    task = r'%s-[%i-%ix%i-%i]-%s-%i-VFs' % (run.name,N_INPUTS, N_UNITS[0], H_LAYERS, N_OUTPUTS, TRAIN_MULTI_DIR.split('/')[-2], count_parameters(model_1))

    output_task = 'outputs/' + TRAIN_MULTI_DIR.split('/')[-2] + '_sbvf_abs_direct'
    output_loss = 'outputs/' + TRAIN_MULTI_DIR.split('/')[-2] + '_sbvf_abs_direct/loss/'
    output_stats = 'outputs/' + TRAIN_MULTI_DIR.split('/')[-2] + '_sbvf_abs_direct/stats/'
    output_models = 'outputs/' + TRAIN_MULTI_DIR.split('/')[-2] + '_sbvf_abs_direct/models/'
    output_val = 'outputs/' + TRAIN_MULTI_DIR.split('/')[-2] + '_sbvf_abs_direct/val/'
    output_logs = 'outputs/' + TRAIN_MULTI_DIR.split('/')[-2] + '_sbvf_abs_direct/logs/'

    directories = [output_task, output_loss, output_stats, output_models, output_val, output_logs]

    for dir in directories:
        try:
            os.makedirs(dir)

        except FileExistsError:
            pass

    history.to_csv(output_loss + task + '.csv', sep=',', encoding='utf-8', header='true')

    plot_history(history, output_loss, True, task)

    torch.save(model_1.state_dict(), output_models + task + '.pt')
    
    #joblib.dump([VFs, W_virt], 'sbvfs.pkl')
    if train_generator.std == True:
        joblib.dump(train_generator.scaler_x, output_models + task + '-scaler_x.pkl')

    if WANDB_LOG:
        # 3️⃣ At the end of training, save the model artifact
        # Name this artifact after the current run
        task_ = r'__%i-%ix%i-%i__%s-direct' % (N_INPUTS, N_UNITS[0], H_LAYERS, N_OUTPUTS, TRAIN_MULTI_DIR.split('/')[-2])
        model_artifact_name = run.id + '_' + run.name + task_
        # Create a new artifact, which is a sample dataset
        model = wandb.Artifact(model_artifact_name, type='model')
        # Add files to the artifact, in this case a simple text file
        model.add_file(local_path=output_models + task + '.pt')
        model.add_file(output_models + task + '-scaler_x.pkl')
        # Log the model to W&B
        run.log_artifact(model)
        # Call finish if you're in a notebook, to mark the run as done
        run.finish()

# -------------------------------
#           Main script
# -------------------------------
if __name__ == '__main__':
    # Creating temporary folder
    try:
        os.makedirs('./temp')
    except FileExistsError:
        pass

    # # Default floating point precision for pytorch
    # torch.set_default_dtype(torch.float64)

    # # Specifying random seed
    # random.seed(SEED)
    # torch.manual_seed(SEED)
    window=Tk()
    #Create a queue to share data between process
    q = multiprocessing.Queue()

    #Create and start the simulation process
    train_ = multiprocessing.Process(None,train,args=(q,))
    train_.start()

    #Create the base plot
    plot()

    #Call a function to update the plot when there is new data
    updateplot(q)

    window.mainloop()

    # # Reading mesh file
    # mesh, connectivity, dof = read_mesh(TRAIN_MULTI_DIR)

    # # Defining geometry limits
    # x_min = min(mesh[:,1])
    # x_max = max(mesh[:,1])
    # y_min = min(mesh[:,-1])
    # y_max = max(mesh[:,-1])

    # # Total degrees of freedom
    # total_dof = mesh.shape[0] * 2

    # # Defining edge boundary conditions
    # bcs = {
    #     'left': {
    #         'cond': [1,0],
    #         'dof': global_dof(mesh[mesh[:,1]==x_min][:,0])},
    #     'bottom': {
    #         'cond': [0,1],
    #         'dof': global_dof(mesh[mesh[:,-1]==y_min][:,0])},
    #     'right': {
    #         'cond': [2,0],
    #         'dof': global_dof(mesh[mesh[:,1]==x_max][:,0])},
    #     'top': {
    #         'cond': [0,0],
    #         'dof': global_dof(mesh[mesh[:,-1]==y_max][:,0])}
    # }

    # # Constructing element properties based on mesh info
    # elements = [Element(connectivity[i,:],mesh[connectivity[i,1:]-1,1:],dof[i,:]) for i in tqdm(range(connectivity.shape[0]))]

    # # Assembling global strain-displacement matrices
    # b_glob, b_bar, b_inv, active_dof = global_strain_disp(elements, total_dof, bcs)

    # # Loading data
    # df_list, _ = load_dataframes(TRAIN_MULTI_DIR)

    # # Merging training data
    # data = pd.concat(df_list, axis=0, ignore_index=True)

    # T_PTS = len(set(data['t']))

    # # Performing test/train split
    # partition = {"train": None, "test": None}

    # # if T_PTS==DATA_SAMPLES:
    # # Reorganizing dataset by tag, subsequent grouping by time increment
    # data_by_tag = [df for _, df in data.groupby(['tag'])]
    # random.shuffle(data_by_tag)
    # data_by_t = [[df for _, df in group.groupby(['t'])] for group in data_by_tag]
    # #random.shuffle(data_by_t)
    # data_by_batches = list(itertools.chain(*data_by_t))
    # #random.shuffle(data_by_batches)

    # batch_size = len(data_by_batches[0]) * T_PTS

    # data = pd.concat(data_by_batches).reset_index(drop=True)

    # trials = list(set(data['tag'].values))
    # test_trials = random.sample(trials, math.floor(len(trials)*TEST_SIZE))
    # train_trials = list(set(trials).difference(test_trials))

    # partition['train'] = data[data['tag'].isin(train_trials)].index.tolist()
    # partition['test'] = data[data['tag'].isin(test_trials)].index.tolist()

    # # Selecting model features
    # X, y, f, coord, info = select_features_multi(data)

    # # Preparing data generators for mini-batch training

    # #partition['train'] = data[data['tag']=='m80_b80_x'].index.tolist()
    # train_generator = DataGenerator(X, y, f, coord, info, partition["train"], batch_size, shuffle=False, std=True, t_pts=T_PTS)
    # test_generator = DataGenerator(X, y, f, coord, info, partition['test'], batch_size, shuffle=False, std=False, t_pts=T_PTS)

    # # Model variables
    # N_INPUTS = X.shape[1]
    # N_OUTPUTS = y.shape[1]

    # N_UNITS = [12,6]
    # H_LAYERS = len(N_UNITS)

    # model_1 = NeuralNetwork(N_INPUTS, N_OUTPUTS, N_UNITS, H_LAYERS)

    # model_1.apply(init_weights)

    # # Training variables
    # epochs = 50000

    # # Optimization variables
    # learning_rate = 0.1
    # loss_fn = sbvf_loss
    # f_loss = torch.nn.MSELoss()

    # optimizer = torch.optim.Adam(params=list(model_1.parameters()), lr=learning_rate)
    # #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50, factor=0.25, min_lr=1e-5)
    # #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=20, T_mult=1, eta_min=0.001)

    # #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,0.99)

    # #constraints=weightConstraint(cond='plastic')

    # # Container variables for history purposes
    # train_loss = []
    # v_work = []
    # val_loss = []
    # epochs_ = []
    # # Initializing the early_stopping object
    # early_stopping = EarlyStopping(patience=2000, path='temp/checkpoint.pt', verbose=True)

    # #wandb.watch(model_1)
    # VFs_train = {key: {k: dict.fromkeys(set(info['inc'])) for k in ['u','e','v_u']} for key in train_trials}
    # VFs_test = {key: {k: dict.fromkeys(set(info['inc'])) for k in ['u','e','v_u']} for key in test_trials}

    # w_virt = {key: {k: dict.fromkeys(set(info['inc'])) for k in ['w_int','w_int_real','w_ext']} for key in set(info['tag'])}

    # upd_tol = 1e-1
    # upd_tol_tune = 3.1
    # vf_updt = False
    # corr_factor = 0
    # isTraining = False

    # for t in range(epochs):

    #     print('\r--------------------\nEpoch [%d/%d]' % (t + 1, epochs))

    #     epochs_.append(t+1)

    #     #start_epoch = time.time()

    #     #Shuffling batches
    #     train_generator.on_epoch_end()
    #     test_generator.on_epoch_end()

    #     #--------------------------------------------------------------
    #     #Calculating VFs
    #     if t==0:
    #         print('Computing virtual fields...')
    #         get_sbvfs(model_1, train_generator)
    #         get_sbvfs(model_1, test_generator, isTrain=False)

    #     #--------------------------------------------------------------

    #     # Train loop
    #     start_train = time.time()

    #     #--------------------------------------------------------------
    #     batch_losses, batch_v_work = train_loop(train_generator, model_1, loss_fn, optimizer)
    #     #--------------------------------------------------------------

    #     #batch_losses, batch_v_work, v_disp, v_strain, v_u = train_loop(train_generator, model_1, loss_fn, optimizer)

    #     train_loss.append(torch.mean(batch_losses).item())
    #     v_work.append(torch.mean(batch_v_work).item())

    #     end_train = time.time()

    #     #Apply learning rate scheduling if defined
    #     try:
    #         scheduler.step(train_loss[t])
    #         #scheduler.step()
    #         print('. t_loss: %.6e -> lr: %.3e // [v_work] -> %.3e -- %.3fs' % (train_loss[t], scheduler._last_lr[0], v_work[t], end_train - start_train))
    #     except:
    #         print('. t_loss: %.6e // [v_work] -> %.3e -- %.3fs' % (train_loss[t], v_work[t], end_train - start_train))

    #     # Test loop
    #     start_test = time.time()

    #     # batch_val_losses = test_loop(test_generator, model_1, loss_fn, v_disp, v_strain)

    #     #-----------------------------------------------------------------------------------
    #     batch_val_losses = test_loop(test_generator, model_1, loss_fn)
    #     #-----------------------------------------------------------------------------------

    #     val_loss.append(torch.mean(batch_val_losses).item())

    #     end_test = time.time()

    #     print('. v_loss: %.6e -- %.3fs' % (val_loss[t], end_test - start_test))

    #     # end_epoch = time.time()

    #     if t > 100:
    #         # Check validation loss for early stopping
    #         early_stopping(val_loss[t], model_1)

    #     optimality = abs(train_loss[t]-train_loss[t-1])
    #     if t!= 0 and t % 10 == 0:
    #         print('\nComputing new virtual fields -- optimality: %.6e' % (optimality))
    #         get_sbvfs(model_1, train_generator)
    #         get_sbvfs(model_1, test_generator,isTrain=False)
    #     else:
    #         print('\noptimality: %.6e' % (optimality))
    #     if (t!= 0 and optimality < 1e-9) or early_stopping.early_stop:
    #             print("Early stopping")
    #             break

    # print("Done!")

    # # load the last checkpoint with the best model
    # #model_1.load_state_dict(torch.load('temp/checkpoint.pt'))

    # epochs_ = np.reshape(np.array(epochs_), (len(epochs_),1))
    # train_loss = np.reshape(np.array(train_loss), (len(train_loss),1))
    # val_loss = np.reshape(np.array(val_loss), (len(val_loss),1))

    # history = pd.DataFrame(np.concatenate([epochs_, train_loss, val_loss], axis=1), columns=['epoch','loss','val_loss'])


    # task = r'[%i-%ix%i-%i]-%s-%i-VFs' % (N_INPUTS, N_UNITS[0], H_LAYERS, N_OUTPUTS, TRAIN_MULTI_DIR.split('/')[-2], count_parameters(model_1))

    # output_task = 'outputs/' + TRAIN_MULTI_DIR.split('/')[-2] + '_sbvf_abs'
    # output_loss = 'outputs/' + TRAIN_MULTI_DIR.split('/')[-2] + '_sbvf_abs/loss/'
    # output_stats = 'outputs/' + TRAIN_MULTI_DIR.split('/')[-2] + '_sbvf_abs/stats/'
    # output_models = 'outputs/' + TRAIN_MULTI_DIR.split('/')[-2] + '_sbvf_abs/models/'
    # output_val = 'outputs/' + TRAIN_MULTI_DIR.split('/')[-2] + '_sbvf_abs/val/'
    # output_logs = 'outputs/' + TRAIN_MULTI_DIR.split('/')[-2] + '_sbvf_abs/logs/'

    # directories = [output_task, output_loss, output_stats, output_models, output_val, output_logs]

    # for dir in directories:
    #     try:
    #         os.makedirs(dir)

    #     except FileExistsError:
    #         pass

    # history.to_csv(output_loss + task + '.csv', sep=',', encoding='utf-8', header='true')

    # plot_history(history, output_loss, True, task)

    # torch.save(model_1.state_dict(), output_models + task + '.pt')
    # np.save(output_logs + 'sbvfs.npy', VFs_train)
    # np.save(output_logs + 'w_virt.npy', w_virt)
    # #joblib.dump([VFs, W_virt], 'sbvfs.pkl')
    # if train_generator.std == True:
    #     joblib.dump(train_generator.scaler_x, output_models + task + '-scaler_x.pkl')

    # Deleting temp folder
    try:
        shutil.rmtree('temp/')
    except FileNotFoundError:
        pass

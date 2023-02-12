from turtle import forward
from pytools import T
from sklearn import preprocessing
import pandas as pd
import os
import numpy as np
import copy
import matplotlib.pyplot as plt
import constants
import copy
from constants import FORMAT_PBAR, LOOK_BACK
import torch
from tqdm import tqdm
from torch import device, dropout, nn
from io import StringIO
import math
import torch.nn.functional as F
from torch.nn.utils import (
  parameters_to_vector as Params2Vec,
  vector_to_parameters as Vec2Params
)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter
import dask.dataframe as dd
import gc
import pyarrow.parquet as pq


# -------------------------------
#        Class definitions
# -------------------------------
def draw_graph(start, watch=[]):
    from graphviz import Digraph

    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    graph = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))

    assert(hasattr(start, "grad_fn"))
    if start.grad_fn is not None:
        _draw_graph(start.grad_fn, graph, watch=watch)

    size_per_element = 0.15
    min_size = 12

    # Get the approximate number of nodes and edges
    num_rows = len(graph.body)
    content_size = num_rows * size_per_element
    size = max(min_size, content_size)
    size_str = str(size) + "," + str(size)
    graph.graph_attr.update(size=size_str)
    graph.render(filename='net_graph.jpg')


def _draw_graph(var, graph, watch=[], seen=[], indent="", pobj=None):
    ''' recursive function going through the hierarchical graph printing off
    what we need to see what autograd is doing.'''
    from rich import print
    
    if hasattr(var, "next_functions"):
        for fun in var.next_functions:
            joy = fun[0]
            if joy is not None:
                if joy not in seen:
                    label = str(type(joy)).replace(
                        "class", "").replace("'", "").replace(" ", "")
                    label_graph = label
                    colour_graph = ""
                    seen.append(joy)

                    if hasattr(joy, 'variable'):
                        happy = joy.variable
                        if happy.is_leaf:
                            label += " \U0001F343"
                            colour_graph = "green"

                            for (name, obj) in watch:
                                if obj is happy:
                                    label += " \U000023E9 " + \
                                        "[b][u][color=#FF00FF]" + name + \
                                        "[/color][/u][/b]"
                                    label_graph += name
                                    
                                    colour_graph = "blue"
                                    break

                            vv = [str(obj.shape[x])
                                  for x in range(len(obj.shape))]
                            label += " [["
                            label += ', '.join(vv)
                            label += "]]"
                            label += " " + str(happy.var())

                    graph.node(str(joy), label_graph, fillcolor=colour_graph)
                    print(indent + label)
                    _draw_graph(joy, graph, watch, seen, indent + ".", joy)
                    if pobj is not None:
                        graph.edge(str(pobj), str(joy))

class weightConstraint(object):
    def __init__(self, cond='plastic'):
        self.cond = cond
        self.count = 0
    def __call__(self, module):

        if hasattr(module,'weight'):

            if (self.cond == 'plastic'):
            
                w=module.weight.data
                w=w.clamp(0.0)
                module.weight.data=w 
                self.count += 1

            else:
                w=module.weight.data
                w=w.clamp(0.0)
                w[:2,-1]=w[:2,-1].clamp(0.0,0.0)
                w[-1,:2]=w[:2,-1].clamp(0.0,0.0)
                module.weight.data=w 
# class LinearExponential(nn.Module):
#     def __init__(self,beta=2.0):
#         super(LinearExponential,self).__init__()
#         self.beta = torch.nn.Parameter(torch.tensor(beta))

#     def forward(self,x):

#         if self.beta == 0.0:
#             return x
#         else: 
#             return (1/(2*torch.pow(self.beta,4)))*torch.square(torch.log10(1+torch.exp(torch.square(self.beta)*x)))
class SoftplusSquared(nn.Module):

    def __init__(self, alpha: float = 0.5, device=None, dtype=None) -> None:
       
        super(SoftplusSquared, self).__init__()
        #self.weight = nn.Parameter(torch.empty(1, **factory_kwargs).fill_(init))
        self.alpha = torch.empty(1).fill_(alpha)
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        #if self.weight == 0.0:
        return 1/(2*torch.pow(self.alpha,4)) * torch.square(torch.log10(1+torch.exp(torch.square(self.alpha)*input)))
        # else:
        #     return 1/(2*torch.pow(self.weight,4)) * torch.square(torch.log10(1+torch.exp(torch.square(self.weight)*input)))

class LinearExponential(nn.Module):
    r"""Applies the element-wise function:
    .. math::
        \text{ELU}(x) = \max(0,x) + \min(0, \alpha * (\exp(x) - 1))
    Args:
        alpha: the :math:`\alpha` value for the ELU formulation. Default: 1.0
        inplace: can optionally do the operation in-place. Default: ``False``
    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input
    .. image:: ../scripts/activation_images/ELU.png
    Examples::
        >>> m = nn.ELU()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """
    __constants__ = ['alpha', 'inplace']
    alpha: float
    inplace: bool

    def __init__(self, alpha: float = 0.1, inplace: bool = False) -> None:
        super(LinearExponential, self).__init__()
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.alpha * input) - 1

    def extra_repr(self) -> str:
        inplace_str = ', inplace=True' if self.inplace else ''
        return 'alpha={}{}'.format(self.alpha, inplace_str)

class QuadraticExponential(nn.Module):
    r"""Applies the element-wise function:
    .. math::
        \text{ELU}(x) = \max(0,x) + \min(0, \alpha * (\exp(x) - 1))
    Args:
        alpha: the :math:`\alpha` value for the ELU formulation. Default: 1.0
        inplace: can optionally do the operation in-place. Default: ``False``
    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input
    .. image:: ../scripts/activation_images/ELU.png
    Examples::
        >>> m = nn.ELU()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """
    __constants__ = ['alpha', 'inplace']
    alpha: float
    inplace: bool

    def __init__(self, alpha: float = 0.05, inplace: bool = False) -> None:
        super(QuadraticExponential, self).__init__()
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.alpha * torch.square(input)) - 1

    def extra_repr(self) -> str:
        inplace_str = ', inplace=True' if self.inplace else ''
        return 'alpha={}{}'.format(self.alpha, inplace_str)

class SoftplusLayer(nn.Module):
    r"""Applies a softplus transformation to the incoming data

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(*, H_{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_{in} = \text{in\_features}`.
        - Output: :math:`(*, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(SoftplusLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, F.softplus(self.weight), self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class InputConvexNN(nn.Module):
    def __init__(self,input_size, output_size, hidden_size, n_hidden_layers=1) -> None:
        super(InputConvexNN, self).__init__()

        self.input_size = input_size
        self.hidden_size  = hidden_size
        self.n_hidden_layers = n_hidden_layers
        self.output_size = output_size

        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.passthrough = nn.ModuleList()

        for i in range(self.n_hidden_layers):
            if i == 0:
                in_ = self.input_size
                out_ = self.hidden_size[i]
                self.layers.append(torch.nn.Linear(in_, out_, bias=True))
            else:
                in_ = self.hidden_size[i-1]
                out_= self.hidden_size[i]

                self.layers.append(SoftplusLayer(in_, out_, bias=True))
                self.passthrough.append(torch.nn.Linear(self.input_size,out_,bias=False))

            self.activations.append(torch.nn.Softplus())
        
        self.layers.append(SoftplusLayer(self.hidden_size[-1], self.output_size, bias=True))
        self.passthrough.append(torch.nn.Linear(self.input_size,self.output_size,bias=False))

    def forward(self,x):
        
        for i,layer in enumerate(self.layers[:-1]):
            
            if i == 0:
                xx = self.activations[i](layer(x))
            else:
                xx = self.activations[i](layer(xx))

        return self.layers[-1](xx)

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_hidden_layers=1,b_norm=False):
        super(NeuralNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size  = hidden_size
        self.n_hidden_layers = n_hidden_layers
        self.output_size = output_size

        self.b_norm = b_norm

        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        #self.dropouts = nn.ModuleList()
        
        if self.b_norm:
            self.b_norms = nn.ModuleList()
        #self.parametrizations = nn.ModuleList()
        #self.drop = nn.Dropout(0.05)

        if self.n_hidden_layers == 0:
            self.layers.append(torch.nn.Linear(self.input_size,self.output_size,bias=True))
            #self.activation = torch.nn.PReLU()
        else:   
            for i in range(self.n_hidden_layers):
                if i == 0:
                    in_ = self.input_size
                    out_ = self.hidden_size[i]
                    
                else:
                    in_ = self.hidden_size[i-1]
                    out_= self.hidden_size[i]

                if self.b_norm:    
                    self.b_norms.append(torch.nn.BatchNorm1d(out_,eps=0.1))
                
                self.layers.append(torch.nn.Linear(in_, out_, bias=True))
                
                self.activations.append(torch.nn.ELU())

                #self.dropouts.append(torch.nn.Dropout(0.05))

                
            self.layers.append(torch.nn.Linear(self.hidden_size[-1], self.output_size, bias=True))

    def forward(self, x):

        if self.n_hidden_layers == 0:

            return self.layers[0](x)

        else:
            for i,layer in enumerate(self.layers[:-1]):
                
                if self.b_norm:
                    x = self.activations[i](self.b_norms[i](layer(x)))
                else:
                    x = self.activations[i](layer(x))
                   
            return self.layers[-1](x)

# # LSTMModel class as in https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_lstm_neuralnetwork/
# class LSTMModel(nn.Module):
#     def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
#         super(LSTMModel, self).__init__()
#         # Hidden dimensions
#         self.device = device
#         self.hidden_dim = hidden_dim

#         # Number of hidden layers
#         self.layer_dim = layer_dim

#         # Building LSTM
#         # batch_first=True causes input/output tensors to be of shape
#         # (batch_dim, seq_dim, feature_dim)
#         self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
#         # Readout layer
#         self.fc = nn.Linear(hidden_dim, output_dim)

#     def forward(self, x):
        
#         h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().cuda()
#         # Initialize cell state
#         c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().cuda()
#         # 28 time steps
#         # We need to detach as we are doing truncated backpropagation through time (BPTT)
#         # If we don't, we'll backprop all the way to the start even after going through another batch
#         out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

#         # Index hidden state of last time step
#         # out.size() --> 100, 28, 100
#         # out[:, -1, :] --> 100, 100 --> just want last time step hidden states! 
#         out = self.fc(out[:, -1, :]) 
#         # out.size() --> 100, 10
#         return out

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, drop_prob=0.2):
        super(GRUModel, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim

        # GRU layers
        self.gru = nn.GRU(input_dim, hidden_dim, layer_dim, batch_first=True, dropout=drop_prob)

        #self.fc_layers = nn.ModuleList()
        # Fully connected layer
        
        self.fc = nn.Linear(hidden_dim, output_dim)
        # self.fc_layers.append(nn.Linear(hidden_dim, hidden_dim))
        # self.fc_layers.append(nn.Linear(hidden_dim, hidden_dim))
        # self.fc_layers.append(nn.Linear(hidden_dim, output_dim))

        self.relu = nn.ReLU()

    def init_hidden(self, batch_size, device=None):
        # Initializing hidden state for first input with zeros
        if device != None:
            self.h0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim).requires_grad_().to(device)
        else:
            self.h0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim).requires_grad_()
        # weight = next(self.parameters()).data
        # self.h0 = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device)

    def forward(self, x):

        # Forward propagation by passing in the input and hidden state into the model
        out, _ = self.gru(x, self.h0.detach())

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(self.relu(out))
        # for layer in self.fc_layers[:-1]:
        #     out = self.relu(layer(out))

        # return self.fc_layers[-1](out)
        return out

# EarlyStopping class as in: https://github.com/Bjarten/early-stopping-pytorch/
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.5e} --> {val_loss:.5e}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

class Element():
    def __init__(self, connect, node_coord, dof) -> None:
        self.id = connect[0]
        self.connect = connect[1:]
        self.node_coord = torch.tensor(node_coord)
        self.global_dof = dof
        self.local_dof = np.arange(len(dof))
    
    def b_el(self, csi=0, eta=0):

        x = [-1,1,1,-1]
        y = [-1,-1,1,1]

        n = list(zip(x,y))

        dN_dcsi = torch.tensor([0.25*(i)*(1+eta*j) for (i,j) in n])
        dN_deta = torch.tensor([0.25*(1+csi*i)*(j) for (i,j) in n])

        dN_csi_eta = torch.stack([dN_dcsi,dN_deta],0)
        J = dN_csi_eta @ self.node_coord

        dN_x_y = torch.linalg.solve(J,dN_csi_eta)

        b_el = torch.zeros(3,len(self.local_dof))

        x_dof = self.local_dof[::2]
        y_dof = self.local_dof[1::2]

        b_el[0,x_dof] += dN_x_y[0,:]
        b_el[1,y_dof] += dN_x_y[1,:]

        b_el[2,x_dof] += dN_x_y[1,:]
        b_el[2,y_dof] += dN_x_y[0,:]

        return b_el

class SBVFLoss(nn.Module):

    def __init__(self, scale_par=0.3, res_scale=False):
        super(SBVFLoss, self).__init__()
        self.scale_par = scale_par
        self.res_scale = res_scale
    
    def forward(self, wi, we):

        res = wi - we

        if self.res_scale:
            ivw_sort = torch.sort(torch.abs(res.detach()),1,descending=True).values
            #ivw_sort = torch.max(torch.abs(res.detach()),1).values
        else:
            ivw_sort = torch.sort(torch.abs(wi.detach()),1,descending=True).values
            #ivw_sort = torch.max(torch.abs(we.detach()),1).values
        
        numSteps = math.floor(self.scale_par * wi.shape[1])

        if numSteps == 0:
            numSteps = 1

        alpha = (1/torch.mean(ivw_sort[:,0:numSteps],1))
        #alpha = (1/ivw_sort)
       
        return torch.sum(torch.square(alpha)*torch.sum(torch.square(res),1))
        #return torch.sum(0.5*torch.square(alpha)*torch.mean(torch.square(res),1))

class BaseLoss(nn.modules.Module):

    def __init__(self, device=None, n_losses=1):
        super(BaseLoss, self).__init__()
        self.device = device

        self.train = False

        #self.mse = torch.nn.MSELoss()

        # Record the weights.
        self.n_losses = n_losses
        self.alphas = torch.zeros((self.n_losses,), requires_grad=False).type(torch.FloatTensor).to(self.device)
    
    def to_eval(self):
        self.train = False

    def to_train(self):
        self.train = True
    
    def mse_constraint(self, res):
        return torch.sum(torch.square(res))/(2*res.shape[0])

    def mse(self, pred, targ):
        res = pred-targ
        return torch.sum(torch.square(res))/(2*res.shape[0])
    
    def forward(self, res_tuples):
        
        loss = [self.mse(*res_tuples[i]) if len(res_tuples[i])>1 else self.mse_constraint(*res_tuples[i]) for i in range(len(res_tuples))]
        
        return loss
    
class CoVWeightingLoss(BaseLoss):

    """
        Wrapper of the BaseLoss which weighs the losses to the Cov-Weighting method,
        where the statistics are maintained through Welford's algorithm. But now for 32 losses.
    """

    def __init__(self, mean_decay=None, device=None, n_losses=1):
        super(CoVWeightingLoss, self).__init__(device, n_losses)

        # # How to compute the mean statistics: Full mean or decaying mean.
        # self.mean_decay = True if args.mean_sort == 'decay' else False
        # self.mean_decay_param = args.mean_decay_param
        
        self.mean_decay_param = mean_decay
        self.current_iter = -1
        self.alphas = torch.zeros((self.n_losses,), requires_grad=False).type(torch.FloatTensor).to(self.device)
        self.weighted_losses = []
        self.unweighted_losses = []

        # Initialize all running statistics at 0.
        self.running_mean_L = torch.zeros((self.n_losses,), requires_grad=False).type(torch.FloatTensor).to(self.device)
        self.running_mean_l = torch.zeros((self.n_losses,), requires_grad=False).type(torch.FloatTensor).to(self.device)
        self.running_S_l = torch.zeros((self.n_losses,), requires_grad=False).type(torch.FloatTensor).to(self.device)
        self.running_std_l = None

    def forward(self, res_tuples):
        # Retrieve the unweighted losses.
        unweighted_losses = super(CoVWeightingLoss, self).forward(res_tuples)
        
        # Put the losses in a list. Just for computing the weights.
        L = torch.tensor(unweighted_losses, requires_grad=False).to(self.device)

        #self.l_norm = torch.linalg.vector_norm(L)
        self.unweighted_losses = L

        # If we are doing validation, we would like to return an unweighted loss be able
        # to see if we do not overfit on the training set.
        if not self.train:
            return torch.sum(self.alphas * L)

        # Increase the current iteration parameter.
        self.current_iter += 1
        # If we are at the zero-th iteration, set L0 to L. Else use the running mean.
        L0 = L.clone() if self.current_iter == 0 else self.running_mean_L
        # Compute the loss ratios for the current iteration given the current loss L.
        l = L / L0

        # If we are in the first iteration set alphas to all 1/32
        if self.current_iter <= 1:
            self.alphas = torch.ones((self.n_losses,), requires_grad=False).type(torch.FloatTensor).to(self.device) / self.n_losses
        # Else, apply the loss weighting method.
        else:
            ls = self.running_std_l / self.running_mean_l
            self.alphas = ls / torch.sum(ls)

        # Apply Welford's algorithm to keep running means, variances of L,l. But only do this throughout
        # training the model.
        # 1. Compute the decay parameter the computing the mean.
        if self.current_iter == 0:
            mean_param = 0.0
        elif self.current_iter > 0 and self.mean_decay_param != None:
            mean_param = self.mean_decay_param
        else:
            mean_param = (1. - 1 / (self.current_iter + 1))

        # 2. Update the statistics for l
        x_l = l.clone().detach()
        new_mean_l = mean_param * self.running_mean_l + (1 - mean_param) * x_l
        self.running_S_l += (x_l - self.running_mean_l) * (x_l - new_mean_l)
        self.running_mean_l = new_mean_l

        # The variance is S / (t - 1), but we have current_iter = t - 1
        running_variance_l = self.running_S_l / (self.current_iter + 1)
        self.running_std_l = torch.sqrt(running_variance_l + 1e-8)

        # 3. Update the statistics for L
        x_L = L.clone().detach()
        self.running_mean_L = mean_param * self.running_mean_L + (1 - mean_param) * x_L

        # Get the weighted losses and perform a standard back-pass.
        weighted_losses = [self.alphas[i] * unweighted_losses[i] for i in range(len(unweighted_losses))]
        self.weighted_losses = weighted_losses

        loss = sum(weighted_losses)
        
        return loss
# -------------------------------
#       Method definitions
# ------------------------------
def rotate_tensor(t,theta,is_reverse=False):
    '''Applies a rotation transformation to a given tensor

    Args:
        t (float): The tensor, or batch of tensors, to be transformed
        theta (float): The angle, or angles, of rotation
        is_reverse (bool): Controls if forward or reverse transformation is applied (default is False)

    Returns:
        t_: the rotated tensor
    '''

    r = np.zeros_like(t)
    r[:,0,0] = np.cos(theta)
    r[:,0,1] = np.sin(theta)
    r[:,1,0] = -np.sin(theta)
    r[:,1,1] = np.cos(theta)
    
    if is_reverse:
        t_ = np.transpose(r,(0,2,1)) @ t @ r
    else:
        t_ = r @ t @ np.transpose(r,(0,2,1))
    
    return t_

def get_principal(var, angles=None):  

    if angles is None:
        # Getting principal angles
        angles = 0.5 * np.arctan(2*var[:,-1] / (var[:,0]-var[:,1]))
        angles[np.isnan(angles)] = 0.0

    # Constructing tensors
    tril_indices = np.tril_indices(n=2)
    var_mat = np.zeros((var.shape[0],2,2))

    var[:,[1,2]] = var[:,[2,1]]

    var_mat[:,tril_indices[0],tril_indices[1]] = var[:,:]
    var_mat[:,tril_indices[1],tril_indices[0]] = var[:,:]

    # Rotating tensor to the principal plane
    var_princ_mat = rotate_tensor(var_mat, angles)
    var_princ_mat[abs(var_princ_mat)<=1e-16] = 0.0
    var_princ = var_princ_mat[:,tril_indices[0][:-1],np.array([0,1,0])[:-1]]

    return var_princ, angles

def stress_from_cholesky(pred, d_e, t_pts, n_elems, n_tests=1, sbvf_gen=False):

    l = torch.reshape(pred,[n_tests, t_pts, n_elems, pred.shape[-1]])
    L = torch.zeros([n_tests, t_pts, n_elems, 3, 3])

    tril_indices = torch.tril_indices(row=3, col=3, offset=0)
    L[: ,: , :, tril_indices[0], tril_indices[1]] = l[:,:]
    H = L @ torch.transpose(L,3,4)

    d_s = (H @ d_e.reshape([n_tests, t_pts , n_elems, d_e.shape[-1], 1])).squeeze(-1)
    
    if n_tests==1:
        s = torch.cumsum(d_s.squeeze(),0).reshape([-1,3])
        L = L.squeeze()
        H = H.squeeze()
  
    else:
        s = torch.cumsum(d_s,0).reshape([-1,3])

    if sbvf_gen:
        return s
    else:
        return s, L, H

def layer_wise_lr(model, lr_mult=0.99, learning_rate=0.1):
    layer_names = []
    for n,p in model.named_parameters():
        if 'b_norms' not in n:
            layer_names.append(n)

    layer_names.reverse()

    parameters = []
    prev_group_name = '.'.join(layer_names[0].split('.')[:2])

    # store params & learning rates
    for idx, name in enumerate(layer_names):

        # parameter group name
        cur_group_name = '.'.join(name.split('.')[:2])

        # update learning rate
        if cur_group_name != prev_group_name:
            learning_rate *= lr_mult
        prev_group_name = cur_group_name

        # display info
        print(f'{idx}: lr = {learning_rate:.6f}, {name}')

        # append layer parameters
        parameters += [{'params': [p for n, p in model.named_parameters() if n == name and p.requires_grad],
                        'lr':learning_rate}]
    
    return parameters


@torch.no_grad()
def plot_grad_flow(named_params, path):

    avg_grads, max_grads, layers = [], [], []
    plt.figure(figsize = ((10,20)))
    
    for n, p in named_params:
        
        if (p.requires_grad) and ('bias' not in n):
            
            layers.append(n)
            avg_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
        
    plt.bar(np.arange(len(max_grads)), max_grads, alpha = 0.1, lw = 1, color = 'c')
    plt.bar(np.arange(len(max_grads)), avg_grads, alpha = 0.1, lw = 1, color = 'b')
    plt.hlines(0, 0, len(avg_grads) + 1, lw = 2, color = 'k')
    plt.xticks(range(0, len(avg_grads), 1), layers, rotation = 'vertical')
    plt.xlim(left = 0, right = len(avg_grads))
    plt.ylim(bottom = -0.001, top = 0.02) #Zoom into the lower gradient regions
    plt.xlabel('Layers')
    plt.ylabel('Average Gradients')
    plt.title('Gradient Flow')
    plt.grid(True)
    plt.legend([
        Line2D([0], [0], color = 'c', lw = 4),
        Line2D([0], [0], color = 'b', lw = 4),
        Line2D([0], [0], color = 'k', lw = 4),
        ],
        ['max-gradient', 'mean-gradient','zero-gradient'])
    
    plt.savefig(path)
    plt.close()

def param_deltas(model):
    
    #n_layers = len(model.layers)
    
    #exceptions = [k for k, v in model.state_dict().items() if ('layers.%i' % (n_layers-1) in k and n_layers > 2) or 'activations' in k or 'b_norms' in k]
    exceptions = [k for k, v in model.state_dict().items() if 'activations' in k or 'b_norms' in k or 'parametrizations'in k or 'passthrough' in k or 'bias' in k]
    
    total_params = sum(p.numel() for k, p in model.state_dict().items() if k not in exceptions)
    
    model_dict = {k: v.repeat([v.numel(),1,1]) for k, v in model.state_dict().items() if k not in exceptions}
    pert_dict = {k: torch.zeros_like(v) for k,v in model_dict.items()}
    
    eval_dicts = [model.state_dict() for p in range(total_params)]

    for k, v in model_dict.items():
        a,b = torch.meshgrid(torch.arange(v.shape[1]),torch.arange(v.shape[2]))
        pert_dict[k][torch.arange(v.shape[0]),a.flatten(),b.flatten()] = -0.1
        model_dict[k] += model_dict[k] * pert_dict[k]

    idx_old = 0
    for k,v in model_dict.items():
        idx = v.shape[0]
        [eval_dicts[idx_old+i].update({k:v[i].squeeze(0)}) for i in range(idx)]
        idx_old+=idx

    return eval_dicts

def global_strain_disp(elements, total_dofs, bcs):
    
    g_dof = list(range(total_dofs))
    n_pts = len(elements)
    n_comps = 3
    b_glob = torch.zeros([n_comps * n_pts, total_dofs])

    # Assembly of global strain-displacement matrix
    for i, element in enumerate(elements):
        
        b_glob[n_comps*i:n_comps*i + n_comps, element.global_dof-1] += element.b_el()
    
    b_bar = copy.deepcopy(b_glob)

    bc_fixed = []
    bc_slaves = []
    bc_masters = []

    for edge, props in bcs.items():

        edge_dof_x = list(props['dof'][::2]-1)
        edge_dof_y = list(props['dof'][1::2]-1)

        if edge == 'left' or edge == 'bottom':
        
            master_dof = list(props['dof'][0:2]-1)
            slave_dof = list(props['dof'][2:]-1)

        elif edge == 'right' or edge == 'top':

            master_dof = list(props['dof'][-2:]-1)
            slave_dof = list(props['dof'][:-2]-1)

        # Set bc along x-direction
        if props['cond'][0] == 0:
            pass
        elif props['cond'][0] == 1:
            bc_fixed += edge_dof_x
        elif props['cond'][0] == 2:
            b_bar[:, master_dof[0]] += torch.sum(b_bar[:,slave_dof[::2]],1)
            bc_slaves += slave_dof[::2]
            bc_masters.append(master_dof[0])
        
        # Set bc along y-direction
        if props['cond'][1] == 0:
            pass
        elif props['cond'][1] == 1:
            bc_fixed += edge_dof_y
        elif props['cond'][1] == 2:
            b_bar[:, master_dof[1]] += torch.sum(b_bar[:,slave_dof[1::2]],1)
            bc_slaves += slave_dof[1::2]
            bc_masters.append(master_dof[1])

    # Defining the active degrees of freedom
    actDOFs = list(set(g_dof)-set(sum([bc_fixed,bc_slaves],[])))
    
    # Checking for incompatible boundary conditions
    if len(list(set(bc_masters).intersection(bc_fixed)))!=0:
        raise Exception('Incompatible BCs, adjacent boundary conditions cannot be both fixed/uniform').with_traceback()

    # Discarding redundant boundary conditions
    b_bar = b_bar[:,actDOFs]

    # Computing pseudo-inverse strain-displacement matrix
    b_inv = torch.linalg.pinv(b_bar)
    
    return b_glob, b_inv, actDOFs

def prescribe_u(u, bcs):
    U = copy.deepcopy(u)
    v_disp = torch.zeros(u.shape[0],u.shape[1],1,2)
    for edge, props in bcs.items():

        edge_dof_x = list(props['dof'][::2]-1)
        edge_dof_y = list(props['dof'][1::2]-1)

        if edge == 'left' or edge == 'bottom':
        
            master_dof = list(props['dof'][0:2]-1)
            slave_dof = list(props['dof'][2:]-1)

        elif edge == 'right' or edge == 'top':

            master_dof = list(props['dof'][-2:]-1)
            slave_dof = list(props['dof'][:-2]-1)

        # Setting bcs along x_direction
        if props['cond'][0] == 0:
            pass
        elif props['cond'][0] == 1:
            U[:,:,edge_dof_x] = 0
        elif props['cond'][0] == 2:
            U[:,:,slave_dof[::2]] = torch.reshape(U[:,:,master_dof[0]],(U.shape[0],U.shape[1],1,1))
        
            v_disp[:,:,:,0] = torch.mean(U[:,:,edge_dof_x],2)

        # Setting bcs along y_direction
        if props['cond'][1] == 0:
            pass
        elif props['cond'][1] == 1:
            U[:,:,edge_dof_y] = 0
        elif props['cond'][1] == 2:
            U[:,:,slave_dof[1::2]] = torch.reshape(U[:,:,master_dof[1]],(U.shape[0],U.shape[1],1,1))
        
            v_disp[:,:,:,1] = torch.mean(U[:,:,edge_dof_y],2)

    return U, v_disp 

def sbvf_loss(int_work, ext_work):

    vw = torch.abs(int_work.detach())
   
    #ivw_sort = torch.sort(torch.abs(vw.flatten()),descending=True).values
    ivw_sort = torch.sort(vw,1,descending=True).values
    
    #numSteps = math.floor(0.3*len(vw.flatten()))
    numSteps = math.floor(0.3*int_work.shape[1])

    alpha = (1/torch.mean(ivw_sort[:,0:numSteps],1))
    #alpha = (1/torch.mean(ivw_sort[0:numSteps])) * torch.ones(int_work.shape[0])

    return torch.sum(torch.square(alpha)*torch.sum(torch.square(int_work-ext_work),1))
    
def custom_loss(int_work, ext_work):
    
    #return torch.sum(torch.square(y_pred+y_true))
    #return torch.mean(torch.mean(torch.square(y_pred-y_true),1))
    return torch.sum(torch.sum(torch.abs(torch.sum(torch.sum(int_work,-1,keepdim=True),-2)-ext_work),1))
    #return (1/(4*int_work.shape[0]*int_work.shape[1]))*torch.sum(torch.sum(torch.square(torch.sum(torch.sum(int_work,-1,keepdim=True),-2)-ext_work),1)) 

def global_dof(connect):
    return np.array(sum([[2*i-1,2*i] for i in connect],[])).astype(int)

def read_mesh(dir):
    
    def get_substring_index(list, sub):
        return next((s for s in list if sub in s), None)

    inp_file = ''
    for r, d, f in os.walk(dir):
        for file in f:
            if '.inp' in file:
                inp_file = dir+file

    lines = []
    with open(inp_file) as f:
        lines = f.readlines()
    f.close()

    start_part = lines.index(get_substring_index(lines,'*Part, name'))
    end_part = lines.index(get_substring_index(lines,'*End Part'))

    lines = lines[start_part:end_part+1]

    start_mesh = lines.index(get_substring_index(lines,'*Node'))
    end_mesh = lines.index(get_substring_index(lines,'*Element, type'))

    start_connect = end_mesh
    end_connect = lines.index(get_substring_index(lines,'*Nset, nset'))

    mesh = ''.join(lines[start_mesh+1:end_mesh]).replace(' ','').split('\n')
    mesh = pd.read_csv(StringIO('\n'.join(mesh)),names=['node','x','y'])

    connect_str = ''.join(lines[start_connect+1:end_connect]).replace(' ','').split('\n')[:-1]

    elem_nodes = len(connect_str[0].split(','))-1

    connectivity = pd.read_csv(StringIO('\n'.join(connect_str)),names=['id']+['n%i'% i for i in range(elem_nodes)])
    dof = [[int(j) for j in i.split(',')][1:] for i in connect_str]
    dof = np.array([sum([[2*i-1,2*i] for i in a],[]) for a in dof])

    return mesh.values, connectivity.values, dof


def plot_history(history, output, is_custom=None, task=None):
    
    if is_custom == None:
        hist = pd.DataFrame(history.history)
        hist['epoch'] = history.epoch
    else:
        hist = history

    plt.rcParams.update(constants.PARAMS)
   
    plt.figure(figsize=(8,6), constrained_layout = True)
    plt.title(task)
    plt.xlabel('Epoch')
    plt.ylabel(r'Mean Square Error [J\textsuperscript{2}]')
    plt.plot(hist['epoch'], hist['loss'], label='Train Error', color='#4b7394')
    plt.plot(hist['epoch'], hist['val_loss'], label = 'Test Error', color='#6db1e2')
    plt.yscale('log')
    
    plt.legend()
  
    plt.savefig(output + task + '.png', format="png", dpi=600, bbox_inches='tight')

def standardize_data(X, y, f, scaler_x = None, scaler_y = None, scaler_f = None):

    if scaler_x == None: 
        scaler_x = preprocessing.StandardScaler()
        scaler_x.fit(X)
        X = scaler_x.transform(X)
    else:
        X = scaler_x.transform(X)

    if scaler_y == None:
        scaler_y = preprocessing.StandardScaler()
        scaler_y.fit(y)
        y = scaler_y.transform(y)
    else:
        y = scaler_y.transform(y)
    
    if scaler_f == None:        
        scaler_f = preprocessing.StandardScaler()
        scaler_f.fit(f)
        f = scaler_f.transform(f)
    else:    
        f = scaler_f.transform(f)

    return X, y, f, scaler_x, scaler_y, scaler_f

def select_features_multi(df):

    #X = df[sum([['exx_t%i'% (i),'eyy_t%i'% (i),'exy_t%i'% (i)] for i in range(LOOK_BACK,0,-1)],[])]
    #X = df[['exx_t', 'exx_t1', 'eyy_t', 'eyy_t1', 'exy_t', 'exy_t1']]
    #X = df[['exx_dt', 'eyy_dt', 'exy_dt','exx_t', 'eyy_t', 'exy_t']]
    #X = df[['exx_t', 'eyy_t', 'exy_t']]

    #X = df[['exx_dot_dir','eyy_dot_dir','exy_dot_dir','exx_t', 'eyy_t', 'exy_t']]
    X = df[['ep_1_dir','ep_2_dir','dep_1','dep_2','ep_1','ep_2']]
    #X = df[['ep_1','ep_2']]
    
    y = df[['ds1','ds2']]
    f = df[['fxx_t','fyy_t']]
    #y = df[['sxx_t','syy_t','sxy_t']]
    #f = df[['fxx_t', 'fyy_t', 'fxy_t']]
    
    coord = df[['id','area']]
    #info = df[['tag','inc','t','exx_p_dot','eyy_p_dot','exy_p_dot']]
    #info = df[['tag','inc','t','exx_dot','eyy_dot','exy_dot','d_exx','d_eyy','d_exy']]
    info = df[['tag','inc','t','theta_ep','s1','s2','theta_sp']]
    
    return X, y, f, coord, info

def drop_features(df, drop_list):

    new_df = df.drop(drop_list, axis=1)

    return new_df

def add_past_step(var_list, lookback, df):

    new_df = copy.deepcopy(df)

    for i in range(lookback):
        for j, vars in enumerate(var_list):
            t = df[vars].values
            t_past = df[vars].values[:-(i+1)]
            zeros = np.zeros((i+1,len(vars)))
            t_past = np.vstack([zeros, t_past])
                
            past_vars = [s.replace('_t','_t'+str(i+1)) for s in vars]
            t_past = pd.DataFrame(t_past, columns=past_vars)

            new_df = pd.concat([new_df, t_past], axis=1)

    return new_df

def smooth_data(df):

    id = list(set(df['id']))[0]
    
    new_df = copy.deepcopy(df)

    for var in ['syy_t','sxy_t']:
        e_var = var.replace('s','e')
        s = df[var].values 
        e = df[e_var].values
    
        f_sigma_el = 20.0 # Elastic smoothing
        f_sigma_pl = 10.0  # Plastic smoothing
        
        y_smooth_el = gaussian_filter(s,sigma=f_sigma_el)
        y_smooth_pl = gaussian_filter(s,sigma=f_sigma_pl)

        if var=='sxy_t':
            y_smooth = np.concatenate([y_smooth_el[:575],y_smooth_pl[575:]])
        else:
            y_smooth = y_smooth_pl

        # fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(5*5, 6),)

        # ax1.plot(e,s, 'b', label = f'Original-elem {id}')
        # x0,x1 = ax1.get_xlim()
        # y0,y1 = ax1.get_ylim()
        # ax1.set_aspect((x1-x0)/(y1-y0))

        # ax2.plot(e,y_smooth,'g',label='smoothing')
        # x0,x1 = ax2.get_xlim()
        # y0,y1 = ax2.get_ylim()
        # ax2.set_aspect((x1-x0)/(y1-y0))
    
    
        # ax3.plot(e,s,'r',label='smooth+original')
        # ax3.plot(e,y_smooth,'g',label='smooth')
        # x0,x1 = ax3.get_xlim()
        # y0,y1 = ax3.get_ylim()
        # ax3.set_aspect((x1-x0)/(y1-y0))
        
        # plt.title(var.replace('_','-'))
        # plt.legend()
        # plt.tight_layout()
        # plt.show()
        
        new_df[var] = y_smooth    
        
    return new_df

def preprocess_vars(var_list, df):

    new_df = copy.deepcopy(df)
    
    vars = [
        'ep_1','ep_2',
        'dep_1','dep_2',
        'ep_1_dir','ep_2_dir', 
        'theta_ep',
        's1','s2',
        'ds1','ds2',
        'theta_sp',
    ]

    # Strain, stress and time variables
    e = df[var_list[0]].values
    e[:,-1] *= 0.5

    s = df[var_list[1]].values

    t = np.reshape(df['t'].values,(len(df),1))
    dt = np.diff(t,axis=0)

    # Calculating principal strains and stresses
    eps_princ, ep_angles = get_principal(e)
    s_princ, sp_angles = get_principal(s)
    # Principal stress rate
    #dot_s_princ = np.gradient(s_princ,t.reshape(-1),axis=0)
    dot_s_princ = np.diff(s_princ,axis=0)/dt
    
    # Principal strain rate
    #dot_e_princ = np.gradient(eps_princ,t.reshape(-1),axis=0)
    dot_e_princ = np.diff(eps_princ,axis=0)/dt
    
    # Direction of strain rate
    de_princ_dir = dot_e_princ/(np.reshape(np.linalg.norm(dot_e_princ,axis=1),(dot_e_princ.shape[0],1)))

    de_princ_dir = np.vstack((de_princ_dir,np.array([np.NaN,np.NaN])))
    dot_s_princ = np.vstack((dot_s_princ,np.array([np.NaN,np.NaN])))
    dot_e_princ = np.vstack((dot_e_princ,np.array([np.NaN,np.NaN])))
    
    princ_vars = pd.DataFrame(
        np.concatenate([eps_princ,dot_e_princ,de_princ_dir,ep_angles.reshape(-1,1),s_princ,dot_s_princ,sp_angles.reshape(-1,1)],1),
        columns=vars
    )

    new_df = pd.concat([new_df,princ_vars],axis=1)

    return new_df

def to_sequences(dataset, vars, seq_size=1):
    
    new_df = copy.deepcopy(dataset)

    e = dataset[vars[1]].values
    s = dataset[vars[0]].values
    f = dataset[vars[2]].values

    e_ = []
    s_ = []
    f_ = []

    e_names = sum([['exx_t%i'% (i-1),'eyy_t%i'% (i-1),'exy_t%i'% (i-1)] for i in range(seq_size,0,-1)],[])
    s_names = ['sxx_t0','syy_t0','sxy_t0']
    f_names = ['fxx_t0','fyy_t0','fxy_t0']

    for i in range(len(dataset)-seq_size+1):
        #print(i)
        e_.append(e[i:(i+seq_size),:].flatten())
        s_.append(s[i+seq_size-1,:].flatten())
        f_.append(f[i+seq_size-1,:].flatten())
        
    e_ = pd.DataFrame(e_,columns=e_names)
    s_ = pd.DataFrame(s_,columns=s_names)
    f_ = pd.DataFrame(f_,columns=f_names)

    return pd.concat([new_df, e_, s_, f_], axis=1)

def pre_process(df_list):

    var_list = [['exx_t','eyy_t','exy_t'],['sxx_t','syy_t','sxy_t']]
    #var_list = [['sxx_t','syy_t','sxy_t'],['exx_t','eyy_t','exy_t'],['fxx_t','fyy_t','fxy_t']]
    #lookback = 1
    
    new_dfs = []

    # # Drop vars in z-direction and add delta_t
    # for df in df_list:
        
    #     new_df = drop_features(df, ['ezz_t', 'szz_t', 'fzz_t'])
    #     new_dfs.append(new_df)

    if LOOK_BACK > 0:
        # Add past variables
        for i, df in enumerate(tqdm(df_list, desc='Pre-processing data',bar_format=FORMAT_PBAR)):
            #new_dfs[i] = smooth_data(df)
            #new_dfs[i] = add_past_step(var_list, LOOK_BACK, df)
            new_dfs.append(preprocess_vars(var_list, df))
            #new_dfs[i] = to_sequences(df, var_list, LOOK_BACK)

    return new_dfs

def load_dataframes(directory, preproc=True, cols=None):

    file_list = []
    df_list = []

    for r, d, f in os.walk(directory):
        for file in f:
            if ('.csv' or '.parquet' in file) and 'elems' not in file:
                file_list.append(os.path.join(directory, file))

    # Loading training datasets
    #use_cols = ['tag','id','inc','t','area','exx_t','eyy_t','exy_t','sxx_t','syy_t','sxy_t','fxx_t','fyy_t']
    
    if 'crux' in directory:
        #df_list = [pd.read_parquet(file, columns=use_cols) for file in tqdm(file_list,desc='Reading .csv files',bar_format=FORMAT_PBAR)]
        df_list = [pq.ParquetDataset(file).read_pandas(columns=cols).to_pandas() for file in tqdm(file_list,desc='Importing dataset files',bar_format=FORMAT_PBAR)]
    else:
        df_list = [pd.read_csv(file, sep=',', index_col=False, header=0, engine='c') for file in tqdm(file_list,desc='Reading .csv files',bar_format=FORMAT_PBAR)]

    gc.collect()

    if preproc:
        df_list = pre_process(df_list)    

    return df_list, file_list
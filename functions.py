from readline import parse_and_bind
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
from turtle import forward
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
from torch import mode, nn
from io import StringIO
import math
import torch.nn.functional as F
from torch.nn.utils import (
  parameters_to_vector as Params2Vec,
  vector_to_parameters as Vec2Params
)
from torch.autograd import Function
import geotorch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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
    
    def __call__(self, module):

        if hasattr(module,'weight'):
            if (self.cond == 'plastic'):
               
                w=module.weight.data
                w=w.clamp(-5.0,5.0)
                module.weight.data=w 

            else:
                w=module.weight.data
                w=w.clamp(0.0)
                w[:2,-1]=w[:2,-1].clamp(0.0,0.0)
                w[-1,:2]=w[:2,-1].clamp(0.0,0.0)
                module.weight.data=w 

           
        # if self.cond == 'elastic':
        #     if hasattr(module,'weight'):
        #         #print("Entered")
        #         w=module.weight.data
        #         w=w.clamp(0.0)
        #         w[:2,-1]=w[:2,-1].clamp(0.0,0.0)
        #         w[-1,:2]=w[:2,-1].clamp(0.0,0.0)
        #         module.weight.data=w
        # elif self.cond == 'plastic':
        #     if hasattr(module,'weight'):
        #         w=module.weight.data
        #         w=w.clamp(0.0)
        #         module.weight.data=w
class SoftPlusSquared(nn.Module):
    def __init__(self,beta=2.0):
        super(SoftPlusSquared,self).__init__()
        self.beta = torch.nn.Parameter(torch.tensor(beta))

    def forward(self,x):

        if self.beta == 0.0:
            return x
        else: 
            return (1/(2*torch.pow(self.beta,4)))*torch.square(torch.log10(1+torch.exp(torch.square(self.beta)*x)))

class soft_exponential(nn.Module):
    '''
    Implementation of soft exponential activation.
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    Parameters:
        - alpha - trainable parameter
    References:
        - See related paper:
        https://arxiv.org/pdf/1602.01321.pdf
    Examples:
        >>> a1 = soft_exponential(256)
        >>> x = torch.randn(256)
        >>> x = a1(x)
    '''
    def __init__(self, init: float = 0.25):
        '''
        Initialization.
        INPUT:
            - in_features: shape of the input
            - aplha: trainable parameter
            aplha is initialized with zero value by default
        '''
        super(soft_exponential,self).__init__()
        
        self.alpha = nn.Parameter(torch.tensor(1.0).fill_(init))

    def forward(self, x):
        '''
        Forward pass of the function.
        Applies the function to the input elementwise.
        '''
        if (self.alpha == 0.0):
            return x

        if (self.alpha < 0.0):
            return -torch.log(1 - self.alpha * (x + self.alpha)) / self.alpha

        if (self.alpha > 0.0):
            return (torch.exp(self.alpha * x) - 1)/ self.alpha + self.alpha

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
        

# -------------------------------
#       Method definitions
# ------------------------------
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
    X = df[['exx_dot_dir','eyy_dot_dir','exy_dot_dir','exx_dot','eyy_dot','exy_dot','exx_t', 'eyy_t', 'exy_t']]
    
    y = df[['sxx_t','syy_t','sxy_t']]
    f = df[['fxx_t','fyy_t','fxy_t']]
    #y = df[['sxx_t','syy_t','sxy_t']]
    #f = df[['fxx_t', 'fyy_t', 'fxy_t']]
    
    coord = df[['dir','id', 'cent_x', 'cent_y','area']]
    #info = df[['tag','inc','t','exx_p_dot','eyy_p_dot','exy_p_dot']]
    info = df[['tag','inc','t','exx_dot','eyy_dot','exy_dot','d_exx','d_eyy','d_exy']]
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

def get_yield(e):
        
    window = 7
    der2 = savgol_filter(e, window_length=window, polyorder=4, deriv=4)
    peaks, _ = find_peaks(np.abs(der2),prominence=np.percentile(np.abs(der2),50))       
    max_der2 = np.max(np.abs(der2[peaks]))
    #max_der2 = np.max(np.abs(der2))
    large = np.where(np.abs(der2) == max_der2)[0]
    gaps = np.diff(large) > window
    begins = np.insert(large[1:][gaps], 0, large[0])
    ends = np.append(large[:-1][gaps], large[-1])
    yield_pt = ((begins+ends)/2).astype(np.int)
    # plt.plot(der2)
    # plt.plot(peaks,der2[peaks],'og')
    # plt.show()
    
    return yield_pt

def add_strain_decomp(var_list, df):

    new_df = copy.deepcopy(df)
    eps_vars = ['exx_dot_dir','eyy_dot_dir','exy_dot_dir','exx_dot','eyy_dot','exy_dot','d_exx','d_eyy','d_exy']

    e = df[var_list[0]].values
    t = np.reshape(df['t'].values,(len(df),1))

    #yield_pt = get_yield(e[:,0])[0]
    # plt.plot(e[:,0],df['sxx_t'])
    # plt.plot(e[:,0][yield_pt],df['sxx_t'][yield_pt],'or')
    # plt.title(list(set(df['tag'])))
    # plt.show()
    # for i in range(e.shape[-1]):
    #     pt = get_yield(e[:,i])
    #     plt.plot(e[:,i])
    #     plt.plot(pt, e[:,i][pt], 'ro')
    #     plt.show()

    #e_e = np.zeros_like(e)
    #e_e[:yield_pt,:] = e[:yield_pt,:]
    #e_p = e - e_e
    #p = np.cumsum(e_p,axis=0)

    # Strain rate
    e_dot = np.diff(e,axis=0)/np.diff(t,axis=0).repeat(3, axis=1)
    e_dot = np.vstack((np.array([0,0,0]),e_dot))
    # Strain rate direction
    d_e= np.diff(e,axis=0)
    e_dot_dir = d_e/np.reshape(np.linalg.norm(d_e,axis=1),(d_e.shape[0],1))
    #p_dot = np.diff(p,axis=0)/np.diff(t,axis=0).repeat(3, axis=1)
    #e_p_dot = np.diff(e_p,axis=0)/np.diff(t,axis=0).repeat(3, axis=1)

    e_dot_dir = np.vstack((np.array([0,0,0]),e_dot_dir))
    #p_dot = np.vstack((np.array([0,0,0]),p_dot))
    #e_p_dot = np.vstack((np.array([0,0,0]),e_p_dot))
    d_e = np.vstack((np.array([0,0,0]),d_e))
    eps = pd.DataFrame(np.concatenate([e_dot_dir,e_dot,d_e],1),columns=eps_vars)
    new_df = pd.concat([new_df,eps],axis=1)

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

    var_list = [['exx_t','eyy_t','exy_t']]
    #var_list = [['sxx_t','syy_t','sxy_t'],['exx_t','eyy_t','exy_t'],['fxx_t','fyy_t','fxy_t']]
    #lookback = 1
    
    new_dfs = []

    # Drop vars in z-direction and add delta_t
    for df in df_list:
        
        new_df = drop_features(df, ['ezz_t', 'szz_t', 'fzz_t'])
        new_dfs.append(new_df)

    if LOOK_BACK > 0:
        # Add past variables
        for i, df in enumerate(tqdm(new_dfs, desc='Loading and processing data',bar_format=FORMAT_PBAR)):

            new_dfs[i] = add_past_step(var_list, LOOK_BACK, df)
            new_dfs[i] = add_strain_decomp(var_list, new_dfs[i])
            #new_dfs[i] = to_sequences(df, var_list, LOOK_BACK)

    return new_dfs

def load_dataframes(directory):

    file_list = []
    df_list = []

    for r, d, f in os.walk(directory):
        for file in f:
            if '.csv' in file:
                file_list.append(directory + file)

    #headers = ['tag','id','dir','x', 'y', 'area', 't', 'sxx_t', 'syy_t', 'szz_t', 'sxy_t', 'exx_t', 'eyy_t', 'ezz_t', 'exy_t', 'fxx_t', 'fyy_t', 'fzz_t', 'fxy_t']
    #headers = ['tag','id','dir','x', 'y', 't', 'sxx_t', 'syy_t', 'szz_t', 'sxy_t', 'exx_t', 'eyy_t', 'ezz_t', 'exy_t', 'fxx_t', 'fyy_t', 'fzz_t', 'fxy_t']

    # Loading training datasets
    df_list = [pd.read_csv(file, sep=',', index_col=False, header=0) for file in file_list]

    df_list = pre_process(df_list)

    return df_list, file_list
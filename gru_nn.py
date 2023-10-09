from pyexpat import model
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import math
import torch.nn.init as init

class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUCell, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)

    def forward(self, x, hx=None):

        # Inputs:
        #       input: of shape (batch_size, input_size)
        #       hx: of shape (batch_size, hidden_size)
        # Output:
        #       hy: of shape (batch_size, hidden_size)
        if hx is None:
            if x.is_cuda:
                hx = Variable(torch.zeros(x.size(0), self.hidden_size), device=torch.device('cuda'))
            else:
                hx = Variable(torch.zeros(x.size(0), self.hidden_size))

        x_t = self.x2h(x)
        h_t = self.h2h(hx)

        x_reset, x_upd, x_new = x_t.chunk(3, 1)
        h_reset, h_upd, h_new = h_t.chunk(3, 1)

        r_t = torch.sigmoid(x_reset + h_reset)
        z_t = torch.sigmoid(x_upd + h_upd)
        n_t = torch.tanh(x_new + (r_t * h_new))

        h_t = z_t * hx + (1 - z_t) * n_t

        return h_t

class LayerNormGRUCell(torch.nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(LayerNormGRUCell, self).__init__()

        self.hidden_size = hidden_size

        self.ln_i2h = torch.nn.LayerNorm(2 * hidden_size, elementwise_affine=False)
        self.ln_h2h = torch.nn.LayerNorm(2 * hidden_size, elementwise_affine=False)
        
        self.i2h = torch.nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.h2h = torch.nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        
        self.ln_new_x = torch.nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.ln_new_h = torch.nn.LayerNorm(hidden_size, elementwise_affine=False)        
        
        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
    
    def forward(self, x, hx=None):

        if hx is None:
            if x.is_cuda:
                hx = Variable(torch.zeros(x.size(0), self.hidden_size), device=torch.device('cuda'))
            else:
                hx = Variable(torch.zeros(x.size(0), self.hidden_size))

        # Linear mappings
        xt = self.i2h(x)
        ht = self.h2h(hx)

        _, _, x_new = xt.chunk(3,1)
        _, _, h_new = ht.chunk(3,1)

        # xt = torch.cat([x_reset, x_upd],1)
        # ht = torch.cat([h_reset, h_upd],1)

        # Layer norm
        xt = self.ln_i2h(xt[:,:-self.hidden_size])
        ht = self.ln_h2h(ht[:,:-self.hidden_size])

        gates = self.sigmoid(xt + ht)

        r_t, z_t = gates.chunk(2,1)           
        
        n_t = self.tanh(self.ln_new_x(x_new) + (r_t * self.ln_new_h(h_new)))
        
        h_t = z_t * hx + (1 - z_t) * n_t

        # Reshape for compatibility
        h_t = h_t.view(h_t.size(0), -1)
        
        return h_t

class customGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, gru_bias=True, fc_bias=True, layer_norm=False, pre_train=False):
        super(customGRU, self).__init__()
        
        self.input_size = input_dim
        self.hidden_size = hidden_dim if type(hidden_dim) is list else [hidden_dim]
        self.num_layers = layer_dim
        self.output_size = output_dim
        self.layer_norm = layer_norm

        self.gru_bias = gru_bias
        self.fc_bias = fc_bias

        # Checking if GPU is available
        self.cuda_available = torch.cuda.is_available()
        
        # Adding GRU cells
        self.rnn_cell_list = self.setup_cells()

        # Adding FC layers / LayerNorms
        if len(self.hidden_size)>1:
            
            self.fc_layers = nn.ModuleList()
            
            if self.layer_norm:
                self.norm_layers = nn.ModuleList()
            
            for i in range(len(self.hidden_size)-1):
                
                in_ = self.hidden_size[i]
                out_ = self.hidden_size[i+1]
                
                self.fc_layers.append(nn.Linear(in_, out_, bias=self.fc_bias))
                
                if self.layer_norm:
                    self.norm_layers.append(nn.LayerNorm(out_, elementwise_affine=False))

            self.fc_layers.append(nn.Linear(self.hidden_size[-1],  self.output_size))
        
        else:

            self.fc = nn.Linear(self.hidden_size[0],  self.output_size)

        # Defining activation function
        self.relu = nn.LeakyReLU()

        if not pre_train:
            self.reset_parameters()

    def reset_parameters(self):

        for n, w in self.named_parameters():
        
            # if self.np_init:

            #     # Weight initialization as in https://openreview.net/pdf/wVqq536NJiG0qV7mtBNp.pdf
            #     # 'IMPROVING PERFORMANCE OF RECURRENT NEURAL NETWORK WITH RELU NONLINEARITY'

            #     if 'bias' in n:
            #         init.zeros_(w.data)
                
            #     elif 'i2h.weight' in n:
            #         init.normal_(w.data, std=1/self.hidden_size[0])
            #         w.data *= np.sqrt(2) * np.exp(1.2/(np.max([self.hidden_size[0],6])-2.4))

            #     elif 'h2h.weight' in n:
                    
            #         r = torch.empty([3,self.hidden_size[0],self.hidden_size[0]])
            #         init.normal_(r)
            #         a = (1/self.hidden_size[0]) * (r.transpose(2,1) @ r)
            #         e = torch.max(torch.linalg.eigvals(a).real, dim=1).values.unsqueeze(-1).unsqueeze(-1)
            #         w.data = (a / e).reshape([-1,a.shape[-1]])

            #     else:
            #         init.normal_(w.data,std=2/(torch.as_tensor(w.shape).sum()))
                
            # else:

            if ('rnn_cell_list' in n) and (len(w.shape) >= 2):
                init.orthogonal_(w.data)
            elif ('fc_layers'in n) and len(w.shape) >=2:
                init.kaiming_uniform_(w.data)
            elif 'bias' in n:
                #init.zeros_(w.data)
                #torch.nn.init.uniform_(w.data, a=0.1)
                w.data.fill_(0.1)
            else:
                init.normal_(w.data)

    def setup_cells(self):

        args = [self.input_size, self.hidden_size[0], self.gru_bias]
        
        rnn_cell_list = nn.ModuleList()

        if self.layer_norm:
            rnn_cell_list += [LayerNormGRUCell(*args) if i == 0 else LayerNormGRUCell(self.hidden_size[0],*args[1:]) for i in range(self.num_layers)]
        else:
            rnn_cell_list += [GRUCell(*args) if i == 0 else GRUCell(self.hidden_size[0],*args[1:]) for i in range(self.num_layers)]
        
        return rnn_cell_list

    def forward(self, x, hx=None):

        # Input: (batch_size, seqence length, input_size)
        #
        # Output: (batch_size, output_size)
        
        if hx is None:
            if self.cuda_available and x.is_cuda:
                h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size[0], device=torch.device('cuda')))
            else:
                h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size[0]))
        else:
            h0 = hx

        outs = []

        hidden = [h0[layer, :, :] for layer in range(self.num_layers)]

        for t in range(x.size(1)):

            # for layer in range(self.num_layers):

            #     if layer == 0:
            #         hidden_l = self.rnn_cell_list[layer](x[:, t, :], hidden[layer])
            #     else:
            #         hidden_l = self.rnn_cell_list[layer](hidden[layer - 1], hidden[layer])
                
            #     hidden[layer] = hidden_l

            # outs.append(hidden_l)
            hidden = [
                self.rnn_cell_list[layer](x[:, t, :], hidden[layer]) if layer == 0 
                else self.rnn_cell_list[layer](hidden[layer - 1], hidden[layer]) for layer in range(self.num_layers)
            ]
            
            outs.append(hidden[-1])

        # Take last time step
        out = outs[-1].squeeze()

        if len(self.hidden_size) > 1:

            for i, layer in enumerate(self.fc_layers[:-1]):

                out = layer(out)

                if self.layer_norm:
                    out =  self.norm_layers[i](out)

                out = self.relu(out)
                    
            return self.fc_layers[-1](out), hidden

        else:
        
            return self.fc(self.relu(out)), hidden 

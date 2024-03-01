import torch
import math

class SBVFLoss(torch.nn.Module):

    def __init__(self, scale_par=0.3, res_scale=False):
        super(SBVFLoss, self).__init__()
        self.scale_par = scale_par
        self.res_scale = res_scale
    
    def forward(self, wi, we):

        res = wi - we

        # Getting number of elements in tensor
        m = torch.tensor(res.size()).prod()

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

class UDVFLoss(torch.nn.Module):
    def __init__(self, normalize=None, type='L2', reduction='mean'):
        super(UDVFLoss, self).__init__()
        self.normalize = normalize
        self.type_ = type
        self.reduction = reduction

        self.init_loss()

    def init_loss(self):
        if self.type_ == 'L2':
            self.l_mse = torch.nn.MSELoss(reduction=self.reduction)
        else:
            self.l_l1 = torch.nn.L1Loss(reduction=self.reduction)
    
    def forward(self, wi, we):
        
        # Calculating residual
        #res = wi - we

        if self.normalize == 'wint':
            
            # Normalizing the residual by the internal virtual work
            wi_max = torch.max(torch.abs(wi.detach()))
            self.res_scale = wi_max
            #res = res / wi_max
            
        elif self.normalize == 'wext':

            # Normalizing the residual by the external virtual work
            we_max = torch.max(torch.abs(we.detach()))
            self.res_scale = we_max
            #res = res / we_max

        if self.type_ == 'L2':

            # Loss based on the mean squared residuals
            #return (1/m) * torch.sum(torch.square(res)) if self.reduction == 'mean' else torch.sum(torch.square(res))
            return self.l_mse(wi/self.res_scale, we/self.res_scale)
            
        elif self.type_ == 'L1':

            # Loss based on the mean of the absolute value of the residuals
            #return (1/m) * torch.sum(torch.abs(res)) if self.reduction == 'mean' else torch.sum(torch.abs(res))
            return self.l_l1(wi/self.res_scale, we/self.res_scale)
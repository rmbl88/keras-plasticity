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
    
    def forward(self, wi, we):
        
        # Calculating residual
        res = wi - we

        # Getting number of elements in tensor
        m = torch.tensor(res.size()).prod()

        if self.normalize == 'wint':
            
            # Normalizing the residual by the internal virtual work
            wi_max = torch.max(torch.abs(wi.detach()))
            #wi_max = torch.max(wi.detach())
            #wi_min = torch.min(wi.detach())

            #wi_range = (wi_max-wi_min) 


            res = res / wi_max
            #res = res / wi_range
            
        elif self.normalize == 'wext':

            # Normalizing the residual by the external virtual work
            we_max = torch.max(torch.abs(we.detach()))
            
            res = res / we_max

        if self.type_ == 'L2':

            # Loss based on the mean squared residuals
            return (1/(2*m)) * torch.sum(torch.square(res)) if self.reduction == 'mean' else torch.sum(torch.square(res))
            
        elif self.type_ == 'L1':

            # Loss based on the mean of the absolute value of the residuals
            return (1/m) * torch.sum(torch.abs(res)) if self.reduction == 'mean' else torch.sum(torch.abs(res))
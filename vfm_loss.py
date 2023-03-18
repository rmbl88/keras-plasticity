import torch
import math

class SBVFLoss(torch.nn.Module):

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

class UDVFLoss(torch.nn.Module):
    def __init__(self, normalize=None):
        super(UDVFLoss, self).__init__()
        self.normalize = normalize
    
    def forward(self, wi, we):
        
        res = wi - we

        if self.normalize == 'wint':
            wi_max = torch.max(torch.abs(wi.detach()))
            return torch.sum(torch.sum(torch.square(res/wi_max), 1))
        elif self.normalize == 'wext':
            we_max = torch.max(torch.abs(we.detach()))
            return torch.sum(torch.sum(torch.square(res/we_max), 1))
        else:
            return torch.sum(torch.sum(torch.square(res), 1))
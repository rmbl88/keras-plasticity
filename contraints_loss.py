import torch
from torch import nn

class ClausiusDuhem(nn.Module):
    def __init__(self, reduction='mean'):
        super(ClausiusDuhem, self).__init__()

        self.reduction = reduction
    
    def forward(self, s: torch.Tensor, s_ii: torch.Tensor, de: torch.Tensor) -> torch.Tensor:

        c_duhem = 0.5 * ((s - s_ii) * de).sum(dim=-1)
            
        l_cd = 0.5 * (torch.nn.functional.relu(-c_duhem)).square()

        if self.reduction == 'mean':

            return l_cd.mean()
        
        elif self.reduction == 'sum':

            return l_cd.sum()
        
class NormalizationLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(NormalizationLoss, self).__init__()

        self.reduction = reduction

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        l_norm = x.abs()

        if self.reduction == 'mean':

            return l_norm.mean()

        elif self.reduction == 'sum':

            return l_norm.sum()


import torch
from torch import nn

class BaseLoss(nn.Module):
    def __init__(self):
        super(BaseLoss, self).__init__()
        self.running_loss = 0.0
        self.total_samples = 0.0

    def reset_stats(self):
        self.running_loss = 0.0
        self.total_samples = 0.0
    
    def report_loss(self):
        return self.running_loss / self.total_samples

class DataLoss(BaseLoss):
    def __init__(self):
        super(DataLoss, self).__init__()
        self.mse_loss = torch.nn.MSELoss()
    
    def forward(self, pred: torch.Tensor, real: torch.Tensor) -> torch.Tensor:

        l_data = self.mse_loss(pred, real)

        self.total_samples += pred.numel()
        self.running_loss += l_data.detach().item() * pred.numel()

        return l_data

class ClausiusDuhem(BaseLoss):
    def __init__(self):
        super(ClausiusDuhem, self).__init__()
    
    def forward(self, s: torch.Tensor, s_ii: torch.Tensor, de: torch.Tensor) -> torch.Tensor:

        c_duhem = 0.5 * ((s - s_ii) * de).sum(dim=-1)
            
        l_cd = (torch.nn.functional.relu(-c_duhem)).square().mean()

        self.total_samples += c_duhem.numel()
        self.running_loss += l_cd.detach().item() * c_duhem.numel()

        return l_cd
        
class NormalizationLoss(BaseLoss):
    def __init__(self):
        super(NormalizationLoss, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        l_norm = x.square().mean()

        self.total_samples += x.numel()
        self.running_loss += l_norm.detach().item() * x.numel()

        return l_norm
        
class TriaxialityLoss(BaseLoss):
    def __init__(self):
        super(TriaxialityLoss, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        s_h = (x[:,0] + x[:,1]) / 2
        s_vm = (torch.nn.functional.relu(x[:,0].square() - x[:,0] * x[:,1] + x[:,1].square() + 3 * x[:,2].square())).pow(0.5)
        
        triax = s_h / (s_vm + 1e-12)

        l_triax = torch.nn.functional.relu(triax.abs() - 2/3).square().mean()

        self.total_samples += triax.numel()
        self.running_loss += l_triax.detach().item() * triax.numel()

        return l_triax
    
class LodeAngleLoss(BaseLoss):
    def __init__(self):
        super(LodeAngleLoss, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        s_h = torch.zeros_like(x)
        s_h[:,:2] += ((x[:,0] + x[:,1]) / 2).unsqueeze(-1)
        
        s_dev = x - s_h
        s_dev_j3 = s_dev[:,0] * s_dev[:,1] - s_dev[:,-1].square()

        s_vm = (torch.nn.functional.relu(x[:,0].square() - x[:,0] * x[:,1] + x[:,1].square() + 3 * x[:,2].square())).pow(0.5)

        triax = s_h[:,0] / (s_vm + 1e-12)

        theta = (-27/2) * triax * (triax.square() - 1/3)

        vl = torch.cos(3 * theta) - (s_dev_j3 / (s_vm + 1e-12)).pow(3)

        l_vl = torch.nn.functional.relu(-vl.abs() + 1).square().mean()

        self.total_samples += vl.numel()
        self.running_loss += l_vl.detach().item() * vl.numel()

        return l_vl
    
class PlasticPowerLoss(BaseLoss):
    def __init__(self):
        super(PlasticPowerLoss, self).__init__()

    def forward(self, s: torch.Tensor, de_dt: torch.Tensor) -> torch.Tensor:

        wp = (s * de_dt).sum(-1) 

        l_wp = torch.nn.functional.relu(-wp).square().mean()

        self.total_samples += wp.numel()
        self.running_loss += l_wp.detach().item() * wp.numel()

        return l_wp
    
class SurfaceTractionLoss(BaseLoss):
    def __init__(self):
        super(SurfaceTractionLoss, self).__init__()
        self.l = torch.nn.MSELoss()

    def forward(self, x: torch.Tensor, surf_trac: torch.Tensor, trac_surf: dict, n_elems: int, ord_elem: torch.Tensor) -> torch.Tensor: 
        
        ord_x = x.reshape(-1,n_elems, x.shape[-1])[:,ord_elem]
        trac = 0.0

        for j, (k, v) in enumerate(trac_surf.items()):
            if k == 'top':
                i = 1
            elif k == 'right':
                i = 0

            area = torch.from_numpy(v['elem_area'].T).float().to(x.device)
            surf_x = ord_x[:,v['surf_elems'], i]

            l_trac = self.l((surf_x * area).sum(-1), surf_trac[:,i]).sqrt()
            trac += l_trac

            self.running_loss += l_trac.detach().item() * surf_x.numel()
            self.total_samples += surf_x.numel()

        return trac
    
# class StressGradient(BaseLoss):
#     def __init__(self):
#         super(StressGradient, self).__init__()
#         self.l = torch.nn.HuberLoss()

#     def forward(self, s: torch.Tensor, b_glob: torch.Tensor, n_elems: int, ord_elem: torch.Tensor) -> torch.Tensor: 
        
#         s_ordered = torch.gather(s.reshape(-1, n_elems, s.shape[-1]), dim=1, index=ord_elem.unsqueeze(-1).repeat(1, 1, s.shape[-1]))
#         #s_ordered = s.reshape(-1, n_elems, s.shape[-1])[:, ord_elem]
        
#         s_grad = b_glob.T.unsqueeze(0) @ s_ordered.reshape(s_ordered.shape[0],-1,1)

#         #l_s_grad = s_grad.square().mean()
#         l_s_grad = self.l(s_grad, torch.zeros_like(s_grad))

#         self.running_loss += l_s_grad.detach().item() * s_grad.numel()
#         self.total_samples += s_grad.numel()
        
#         return l_s_grad

class UDVFLoss(BaseLoss):
    def __init__(self, normalize=None, type='L2'):
        super(UDVFLoss, self).__init__()
        self.normalize = normalize
        self.type_ = type

        self.init_loss()

    def init_loss(self):
        if self.type_ == 'L2':
            self.l_mse = torch.nn.MSELoss()
        else:
            self.l_l1 = torch.nn.L1Loss()
    
    def forward(self, wi, we):
        
        if self.normalize == 'wint':
            
            # Normalizing the residual by the internal virtual work
            wi_max = torch.max(torch.abs(wi.detach()))
            self.res_scale = wi_max
            
        elif self.normalize == 'wext':

            # Normalizing the residual by the external virtual work
            we_max = torch.max(torch.abs(we.detach()))
            self.res_scale = we_max

        else:
            self.res_scale = 1.0

        if self.type_ == 'L2':

            # Loss based on the mean squared residuals
            l_udvf = self.l_mse(wi, we)
            
        elif self.type_ == 'L1':

            # Loss based on the mean of the absolute value of the residuals
            l_udvf = self.l_l1(wi, we)
        
        self.running_loss += l_udvf.detach().item() * wi.numel()
        self.total_samples += wi.numel()
        
        return l_udvf
    
def instantiate_losses(loss_name):
    if loss_name == "data":
        return DataLoss()
    if loss_name == "vfm":
        return UDVFLoss() 
    elif loss_name == "clausius":
        return ClausiusDuhem()
    elif loss_name == "normalization":
        return NormalizationLoss()
    elif loss_name == 'triax':
        return TriaxialityLoss()
    elif loss_name == 'p_power':
        return PlasticPowerLoss()
    elif loss_name == 'lode':
        return LodeAngleLoss()
    elif loss_name == 'surf_trac':
        return SurfaceTractionLoss()
    elif loss_name == 's_grad':
        return StressGradient()


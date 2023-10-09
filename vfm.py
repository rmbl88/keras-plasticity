from ast import Is
import copy
from signal import default_int_handler
import time
from xml.dom import INDEX_SIZE_ERR
import numpy as np
from math import pi
import torch
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import gc
from tqdm import tqdm
from tqdm import trange

from constants import FORMAT_PBAR,FORMAT_PBAR_SUB

def get_ud_vfs(centroids, surf_nodes, width, height):

    # General specimen dimensions
    W = width
    H = height

    # Element centroid coordinates
    CENT_X = centroids[:,0]
    CENT_Y = centroids[:,1]

    # Nodal coordinates of traction surfaces
    X = surf_nodes[:,0] if type(surf_nodes) is not list else surf_nodes[0]
    Y = surf_nodes[:,1] if type(surf_nodes) is not list else surf_nodes[1]

    # Auxialiary vectors
    ones = np.ones_like(CENT_X)
    zeros = np.zeros_like(CENT_X)

    zeros_ = np.zeros(surf_nodes.shape[0]) if type(surf_nodes) is not list else 0.
    dim = 1 if type(surf_nodes) is not list else 0 


    # Virtual displacement fields
    v_disp = {
        1: np.stack([X / W, zeros_],dim),
        2: np.stack([zeros_, Y / H],dim),
        3: np.stack([np.sin(X * pi / W) * np.sin(Y * pi / H), np.sin(X * pi / W) * np.sin(Y * pi / H)],dim),
        #4: np.stack([(1 / pi) * np.sin(X * pi / (0.5*W)) * np.sin(Y * pi / (0.5*H)), (1 / pi) * np.sin(X * pi / (0.5*W)) * np.sin(Y * pi / (0.5*H))],dim),
        #5: np.stack([(Y / H) * np.sin(X * pi / W), zeros_], dim),
        #6: np.stack([zeros_, (X / W) * np.sin(Y * pi / H)], dim)
       
    }

    # Defining virtual strain fields
    v_strain = {
        1:np.stack([ones / W, zeros, zeros],1),
        2:np.stack([zeros, ones / H, zeros],1),
        3:np.stack([(pi / W) * np.cos(CENT_X * pi / W) * np.sin(CENT_Y * pi / H), (pi / H) * np.sin(CENT_X * pi / W) * np.cos(CENT_Y * pi / H), (pi / H) * np.sin(CENT_X * pi / W) * np.cos(CENT_Y * pi / H) + (pi / W) * np.cos(CENT_X * pi / W) * np.sin(CENT_Y * pi / H)],1),
        #4:np.stack([(2 / W) * np.cos(CENT_X * pi / (0.5*W)) * np.sin(CENT_Y * pi / (0.5*H)), (2 / H) * np.sin(CENT_X * pi / (0.5*W)) * np.cos(CENT_Y * pi / (0.5*H)), (2 / H) * np.sin(CENT_X * pi / (0.5*W)) * np.cos(CENT_Y * pi / (0.5*H)) + (2 / W) * np.cos(CENT_X * pi / (0.5*W)) * np.sin(CENT_Y * pi / (0.5*H))],1),
        #5:np.stack([(CENT_Y*pi/(H*W)) * np.cos(CENT_X*pi/W), zeros, (1/H)*np.sin(CENT_X*pi/W)],1),
        #6:np.stack([zeros, (CENT_X*pi/(W*H)) * np.cos(CENT_Y*pi/W), (1/W) * np.sin(CENT_Y*pi/H)],1)
        
    }

    #mults = [0.5, 0.625, 0.75, 0.875, 1, 1.25, 1.5, 1.625, 1.75, 1.875]

    # for i, m in enumerate(mults):
    #     v_disp[i+1+len(v_disp.keys())] = np.stack([np.sin(X * pi / (m*W)), zeros_], dim)
    #     v_disp[i+2+len(v_disp.keys())] = np.stack([zeros_, np.sin(Y * pi / (m*H))], dim)
    #     v_strain[i+1+len(v_strain.keys())] = np.stack([(pi / (m*W)) * np.cos(CENT_X * pi / (m*W)), zeros, zeros],1)
    #     v_strain[i+2+len(v_strain.keys())] = np.stack([zeros, (pi / (m*H)) * np.cos(CENT_Y * pi / (m*H)), zeros],1)

    # 6: np.stack([np.sin(X * pi / (0.5*W)), zeros_], dim),
    #     7: np.stack([zeros_, np.sin(Y * pi / (0.5*H))], dim),
    #     8: np.stack([np.sin(X * pi / (0.625*W)), zeros_], dim),
    #     9: np.stack([zeros_, np.sin(Y * pi / (0.625*H))], dim),
    #     10: np.stack([np.sin(X * pi / (0.75*W)), zeros_], dim),
    #     11: np.stack([zeros_, np.sin(Y * pi / (0.75*H))], dim),
    #     12: np.stack([np.sin(X * pi / (0.875*W)), zeros_], dim),
    #     13: np.stack([zeros_, np.sin(Y * pi / (0.875*H))], dim),
    #     14: np.stack([np.sin(X * pi / W), zeros_], dim),
    #     15: np.stack([zeros_, np.sin(Y * pi / H)], dim),
    #     16: np.stack([np.sin(X * pi / (1.25*W)), zeros_], dim),
    #     17: np.stack([zeros_, np.sin(Y * pi / (1.25*H))], dim),
    #     18: np.stack([np.sin(X * pi / (1.5*W)), zeros_], dim),
    #     19: np.stack([zeros_, np.sin(Y * pi / (1.5*H))], dim),
    #     20: np.stack([np.sin(X * pi / (1.625*W)), zeros_], dim),
    #     21: np.stack([zeros_, np.sin(Y * pi / (1.625*H))], dim),
    #     22: np.stack([np.sin(X * pi / (1.75*W)), zeros_], dim),
    #     23: np.stack([zeros_, np.sin(Y * pi / (1.75*H))], dim),
    #     24: np.stack([np.sin(X * pi / (1.875*W)), zeros_], dim),
    #     25: np.stack([zeros_, np.sin(Y * pi / (1.875*H))], dim),
    

    
    # Total number of virtual fields
    total_vfs = len(v_disp.keys())

    # Converting virtual displacement/strain fields dictionaries into a tensors
    v_disp = np.stack(list(v_disp.values())).astype(np.float32)
    v_strain = np.stack(list(v_strain.values())).astype(np.float32)

    # Convert to absolute zero
    eps = np.finfo(np.float32).eps

    v_disp[np.abs(v_disp) < eps] = 0
    v_strain[np.abs(v_strain) < eps] = 0

    return total_vfs, v_disp, v_strain

def internal_vw(stress, v_strain, area):

    return torch.sum(torch.sum(stress * v_strain * area, -1, keepdim=True), 1)

def external_vw(force, v_disp):
    
    return torch.sum(force * v_disp, -1, keepdim=True)

def param_deltas(model, dp=0.01):
    
    exceptions = [k for k, v in model.state_dict().items() if 'gru' in k or 'bias' in k]
    
    total_params = sum(p.numel() for k, p in model.state_dict().items() if k not in exceptions)
    
    model_dict = {k: v.repeat([v.numel(),1,1]) for k, v in model.state_dict().items() if k not in exceptions}
    pert_dict = {k: torch.zeros_like(v) for k,v in model_dict.items()}
    
    #eval_dicts = [model.state_dict() for p in range(total_params)]
    eval_dicts = {p: model.state_dict() for p in range(total_params)}
        
    for k, v in model_dict.items():
        a,b = torch.meshgrid(torch.arange(v.shape[1]),torch.arange(v.shape[2]),indexing="ij")
        pert_dict[k][torch.arange(v.shape[0]),a.flatten(),b.flatten()] = -dp
        model_dict[k] += model_dict[k] * pert_dict[k]

    idx_old = 0
    for k,v in model_dict.items():
        idx = v.shape[0]
        [eval_dicts[idx_old+i].update({k:v[i].squeeze(0)}) for i in range(idx)]
        idx_old+=idx

    return eval_dicts

def sigma_deltas(args):

    param_dict, model, x, s, key = args

    x = torch.split(x,x.size(0)//6,0)

    model.eval()
    with torch.no_grad():
        model.load_state_dict(param_dict)
        pred = [model(x[i])[0] for i in range(len(x))] 
    
    d_sigma = (s - torch.cat(pred))
     
    #d_sigma = torch.reshape(d_sigma,s.shape)

    # if inc_vfs:
    #     ##### INCREMENTAL DELTA STRESS #############
    #     #n_elems = s.shape[1] // T_PTS
    #     ds_ = torch.reshape(d_sigma,[d_sigma.shape[0],T_PTS,n_elems,3])
        
    #     dd_sigma = torch.zeros_like(ds_)
    #     dd_sigma[:,1:] = (ds_[:,1:] - ds_[:,:-1]) / 0.02

    #     return dd_sigma.reshape(s.shape)
    #     # # ##############################################
    # else:
    return d_sigma, key

def prescribe_u(u, bcs):

    U = copy.deepcopy(u)
    v_disp = torch.zeros([*u.shape[:2],1,2])
    
    for _, props in bcs['b_conds'].items():

        edge_dof_x = list(props['dof'][::2]-1)
        edge_dof_y = list(props['dof'][1::2]-1)
            
        master_dof = list(props['m_dof']-1)
        slave_dof = list(set(list(props['dof']-1)) - set(master_dof))

        # Setting bcs along x_direction
        if props['cond'][0] == 0:
            pass
        elif props['cond'][0] == 1:
            U[:,:,edge_dof_x] = 0.
        elif props['cond'][0] == 2:
            U[:,:,slave_dof[::2]] = torch.reshape(U[:,:,master_dof[0]],[*U.shape[:2],1,1])
        
            v_disp[:,:,:,0] = torch.mean(U[:,:,edge_dof_x],2)

        # Setting bcs along y_direction
        if props['cond'][1] == 0:
            pass
        elif props['cond'][1] == 1:
            U[:,:,edge_dof_y] = 0.
        elif props['cond'][1] == 2:
            U[:,:,slave_dof[1::2]] = torch.reshape(U[:,:,master_dof[1]],[*U.shape[:2],1,1])
        
            v_disp[:,:,:,1] = torch.mean(U[:,:,edge_dof_y],2)

    return U, v_disp 

def sbv_fields(d_sigma, b_glob, b_inv, t_pts, n_elems, bcs, active_dof):

    n_dof = b_glob.shape[-1]

    # Reshaping d_sigma for least-square system
    d_sigma = d_sigma.reshape([d_sigma.shape[0],n_elems,t_pts,d_sigma.shape[-1]]).permute(0,2,1,3)
    d_s = torch.reshape(d_sigma,[*d_sigma.shape[:2], -1]).unsqueeze(-1)
    

    v_u = torch.zeros([*d_s.shape[:2], n_dof, 1])

    # Computing virtual displacements (all dofs)
    v_u[:, :, active_dof] = torch.matmul(b_inv,d_s)

    # Prescribing displacements
    v_u, v_disp = prescribe_u(v_u, bcs)

    v_strain = torch.matmul(torch.as_tensor(b_glob,dtype=torch.float32),v_u).squeeze(-1)
    v_strain = torch.reshape(v_strain, d_sigma.shape).permute(0,2,1,3)

    return v_u, v_disp, v_strain

def get_sbvfs(model, d_loader, b_glob, b_inv, bcs, active_dof, isTrain = True):
  
    IS_CUDA = next(model.parameters()).is_cuda

    mdl = copy.deepcopy(model)
    
    d_loader.dataset.shuffle = False
    
    param_dicts = param_deltas(mdl)
    
    n_vfs = len(param_dicts)
    
    model.eval()
    with torch.no_grad():

        num_batches = len(d_loader)
        #t_pts = dataloader.t_pts
        #n_elems = batch_size // t_pts

        iterator = iter(d_loader)
        data = [next(iterator) for i in range(num_batches)]

        if IS_CUDA:
            x = {data[i][-1][0]: (data[i][0].squeeze(0).reshape([-1,*data[i][0].shape[-2:]]).cuda(), data[i][6], data[i][7]) for i in range(num_batches)}
        else:
            x = {data[i][-1][0]: (data[i][0].squeeze(0).reshape([-1,*data[i][0].shape[-2:]]), data[i][6], data[i][7]) for i in range(num_batches)}
        
        s = {k: model(v[0])[0].detach() for k,v in x.items()}

    # # z = list(zip(*x.values()))
    # z = list(x.values())
    # eps = torch.stack(z[0])
    # de = torch.stack(z[1])
    # sigma = torch.stack(list(s.values()))

    #cProfile.runctx('map(lambda p_dict: sigma_deltas(p_dict,model=mdl,x=eps,d_e=de,s=sigma), param_dicts)',{'sigma_deltas':sigma_deltas,'mdl':mdl,'eps':eps,'de':de,'sigma':sigma,'param_dicts':param_dicts},{})
    
    v_fields = {k: {'v_u': None, 'v_e': None} for k,_ in x.items()}
    #v_fields = {k: {'u': None, 'v_u': None, 'v_e': None} for k,_ in x.items()}

    for i,(k,v) in enumerate(x.items()):
        
        n_elems = v[-1]
        t_pts = v[-2]

        ds = torch.empty([len(param_dicts.keys()),v[0].size(0),3])
        pbar = tqdm(total=len(param_dicts.keys()), desc=f'Generating virtual fields... {k}', bar_format=FORMAT_PBAR_SUB, leave=False, position=0)
        
        with ThreadPoolExecutor() as p:

            futures = [p.submit(sigma_deltas, (p_dict, mdl, v[0], s[k], key)) for key, p_dict in param_dicts.items()]
            
            for future in as_completed(futures):
                data, vf_key = future.result()
                pbar.update(n=1)
                ds[vf_key] = data.cpu()
    
        p.shutdown()
        
        _, v_disp, v_strain = sbv_fields(ds, b_glob, b_inv.cpu(), t_pts, n_elems, bcs, active_dof)

        #v_fields[k]['u'] = v_u
        v_fields[k]['v_u'] = v_disp
        v_fields[k]['v_e'] = v_strain

    print('\nFinished generating virtual fields...\n')   
    
    return v_fields
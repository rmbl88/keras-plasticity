import numpy as np
from math import pi
import torch

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
        3: np.stack([(1 / pi) * np.sin(X * pi / W) * np.sin(Y * pi / H), (1 / pi) * np.sin(X * pi / W) * np.sin(Y * pi / H)],dim),
        4: np.stack([(1 / pi) * np.sin(X * pi / (0.5*W)) * np.sin(Y * pi / (0.5*H)), (1 / pi) * np.sin(X * pi / (0.5*W)) * np.sin(Y * pi / (0.5*H))],dim),
        # 5: np.stack([(Y / H) * np.sin(X * pi / W), zeros_], dim),
        # 6: np.stack([zeros_, (X / W) * np.sin(Y * pi / H)], dim)
    }

    # Defining virtual strain fields
    v_strain = {
        1:np.stack([ones / W, zeros, zeros],1),
        2:np.stack([zeros, ones / H, zeros],1),
        3:np.stack([(1 / W) * np.cos(CENT_X * pi / W) * np.sin(CENT_Y * pi / H), (1 / H) * np.sin(CENT_X * pi / W) * np.cos(CENT_Y * pi / H), (1 / H) * np.sin(CENT_X * pi / W) * np.cos(CENT_Y * pi / H) + (1 / W) * np.cos(CENT_X * pi / W) * np.sin(CENT_Y * pi / H)],1),
        4:np.stack([(2 / W) * np.cos(CENT_X * pi / (0.5*W)) * np.sin(CENT_Y * pi / (0.5*H)), (2 / H) * np.sin(CENT_X * pi / (0.5*W)) * np.cos(CENT_Y * pi / (0.5*H)), (2 / H) * np.sin(CENT_X * pi / (0.5*W)) * np.cos(CENT_Y * pi / (0.5*H)) + (2 / W) * np.cos(CENT_X * pi / (0.5*W)) * np.sin(CENT_Y * pi / (0.5*H))],1),
        # 5:np.stack([(CENT_Y*pi/(H*W)) * np.cos(CENT_X*pi/W), zeros, (1/H)*np.sin(CENT_X*pi/W)],1),
        # 6:np.stack([zeros, (CENT_X*pi/(W*H)) * np.cos(CENT_Y*pi/W), (1/W) * np.sin(CENT_Y*pi/H)],1)
    }

    # mults = [0.5, 0.625, 0.75, 0.875, 1, 1.25, 1.5, 1.625, 1.75, 1.875]

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
    
    ivw = torch.sum(torch.sum(stress * v_strain * area, -1, keepdim=True), 1)
    
    return ivw

def external_vw(force, v_disp):
   
    evw = torch.sum(force * v_disp, -1, keepdim=True)
    
    return evw
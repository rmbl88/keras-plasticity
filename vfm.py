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
    X = surf_nodes[0]
    Y = surf_nodes[1]

    # Auxialiary vectors
    ones = np.ones_like(CENT_X)
    zeros = np.zeros_like(CENT_X)

    # Virtual displacement fields
    v_disp = {
        1: np.array([X / W, 0.]),
        2: np.array([0., Y / H]),
        3: np.array([np.sin(X * pi / W) * np.sin(Y * pi / H), np.sin(X * pi / W) * np.sin(Y * pi / H)]),
    }    

    # Defining virtual strain fields
    v_strain = {
        1:np.stack([ones / W, zeros, zeros],1),
        2:np.stack([zeros, ones / H, zeros],1),
        3:np.stack([(pi / W) * np.cos(CENT_X * pi / W) * np.sin(CENT_Y * pi / H), (pi / H) * np.sin(CENT_X * pi / W) * np.cos(CENT_Y * pi / H), (pi / H) * np.sin(CENT_X * pi / W) * np.cos(CENT_Y * pi / H) + (pi / W) * np.cos(CENT_X * pi / W) * np.sin(CENT_Y * pi / H)],1)
    }
    
    # Total number of virtual fields
    total_vfs = len(v_disp.keys())

    # Converting virtual displacement/strain fields dictionaries into a tensors
    v_disp = np.stack(list(v_disp.values()))
    v_strain = np.stack(list(v_strain.values()))

    # Convert to absolute zero
    eps = 1e-15

    v_disp[np.abs(v_disp) < eps] = 0
    v_strain[np.abs(v_strain) < eps] = 0

    return total_vfs, v_disp, v_strain

def internal_vw(stress, v_strain, area):
    
    ivw = torch.sum(torch.sum(stress * v_strain * area, -1, keepdim=True), 1)
    
    return ivw

def external_vw(force, v_disp):
   
    evw = torch.sum(force * v_disp, -1, keepdim=True)
    
    return evw
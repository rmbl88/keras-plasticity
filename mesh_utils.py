import torch
import copy
import time
import glob
import os
import numpy as np
import pandas as pd
from io import StringIO

#-------------------------------------------------------------------------------------------------------------------
#                                                 CLASS DEFINITIONS
#-------------------------------------------------------------------------------------------------------------------

class Element():
    """
    A class to represent a bilinear quadrilateral element.

    Attributes
    ----------
        id : int
            element id
        connect : numpy.ndarray
            element's connectivity
        node_coord : numpy.ndarray
            element's nodal coordinates
        global_dof : numpy.ndarray
            element's global degrees of freedom
        local_dof : numpy.ndarray
            elements's local degrees of freedom

    Methods
    -------
        b_el(csi=0, eta=0):
            Computes the shape function derivatives of a bilinear quadrilateral element.
    """
    def __init__(self, connect, node_coord, dof) -> None:
        self.id = connect[0]
        self.connect = connect[1:]
        self.node_coord = node_coord
        self.global_dof = dof
        self.local_dof = np.arange(len(dof))
    
    def b_el(self, csi=0, eta=0):
        '''
        Computes the shape function derivatives of a bilinear quadrilateral element.

        Parameters:
        -----------
            csi : float
                csi coordinate of the natural axes
            eta : float
                eta coordinate of the natural axes
        
        Returns:
        --------
            b_el : numpy.ndarray
                element shape function derivative matrix
        '''

        x = [-1,1,1,-1]
        y = [-1,-1,1,1]

        n = list(zip(x,y))

        dN_dcsi = np.array([0.25*(i)*(1+eta*j) for (i,j) in n])
        dN_deta = np.array([0.25*(1+csi*i)*(j) for (i,j) in n])

        dN_csi_eta = np.vstack([dN_dcsi,dN_deta])
        J = dN_csi_eta @ self.node_coord

        dN_x_y = np.linalg.solve(J,dN_csi_eta)

        b_el = np.zeros([3,len(self.local_dof)])

        x_dof = self.local_dof[::2]
        y_dof = self.local_dof[1::2]

        b_el[0,x_dof] += dN_x_y[0,:]
        b_el[1,y_dof] += dN_x_y[1,:]

        b_el[2,x_dof] += dN_x_y[1,:]
        b_el[2,y_dof] += dN_x_y[0,:]

        return b_el

#-------------------------------------------------------------------------------------------------------------------
#                                                 METHOD DEFINITIONS
#-------------------------------------------------------------------------------------------------------------------

def read_mesh(dir: str):
    '''
    Reads mesh information from an .inp file.

    Parameters:
    -----------
        dir : str
            Path for the .inp file

    Returns:
    --------
        mesh : numpy.ndarray
            Array containing nodal coordinates and flags
        connectivity : numpy.ndarray
            Array containing the element labes and connectivity table
        dof : numpy.ndarray
            Array containing the global degrees of freedom of each element
    '''
    
    # Gets the index of a substring from the lines of a file
    def get_substring_index(list, sub):
        return next((s for s in list if sub in s), None)

    # Importing .inp file
    inp_file = glob.glob(os.path.join(dir, '*.inp'))[0]

    # Reading file lines
    lines = []
    with open(inp_file) as f:
        lines = f.readlines()
    f.close()

    # Locating keywords in file lines
    start_part = lines.index(get_substring_index(lines,'*Part, name'))
    end_part = lines.index(get_substring_index(lines,'*End Part'))

    lines = lines[start_part:end_part+1]

    start_mesh = lines.index(get_substring_index(lines,'*Node'))
    end_mesh = lines.index(get_substring_index(lines,'*Element, type'))

    start_connect = end_mesh
    end_connect = lines.index(get_substring_index(lines,'*Nset, nset'))

    # Nodal coordinates and flags
    mesh = ''.join(lines[start_mesh+1:end_mesh]).replace(' ','').split('\n')
    mesh = pd.read_csv(StringIO('\n'.join(mesh)),names=['node','x','y'])

    # Connectivity info
    connect_str = ''.join(lines[start_connect+1:end_connect]).replace(' ','').split('\n')[:-1]
    elem_nodes = len(connect_str[0].split(','))-1
    connectivity = pd.read_csv(StringIO('\n'.join(connect_str)),names=['id']+['n%i'% i for i in range(elem_nodes)])
    
    # Degrees of freedom
    dof = [[int(j) for j in i.split(',')][1:] for i in connect_str]
    dof = np.array([sum([[2*i-1,2*i] for i in a],[]) for a in dof])

    return mesh.values, connectivity.values, dof

def import_mesh(dir: str):

    mesh, connectivity, _ = read_mesh(dir)

    nodes = mesh[:,1:]
    connectivity = connectivity[:,1:] - 1

    return nodes, connectivity

def global_dof(connect):
    '''
    Gets the global degrees of freedom from a connectivity table.

    Parameters:
    -----------
        connect : array
            connectivity table

    Returns:
    --------
        g_dof : array
            array of global degrees of freedom
    '''

    g_dof = np.array(sum([[2*i-1,2*i] for i in connect],[])).astype(int)

    return g_dof

def get_surf_elems(mesh: list, connectivity: np.ndarray, b_conds: dict, side: str):

    # Defining geometry limits
    x_min, x_max, y_min, y_max = get_geom_limits(mesh)

    elem_idx = connectivity[np.any(np.isin(connectivity[:,1:], b_conds[side]['nodes'][:,0]), axis=1)][:,0]

    elem_coords = [mesh[connectivity[connectivity[:,0]==elem][0][1:]-1,:][:,1:] for elem in elem_idx]

    if side == 'top':
        idx = 1
        idx_area = 0
        surf_coord = y_max
    elif side == 'right':
        idx = 0
        idx_area = 1
        surf_coord = x_max

    elem_area = np.array([abs(np.diff(coord[coord[:,idx] == surf_coord][:,idx_area])) for coord in elem_coords])

    return {'surf_elems': elem_idx-1, 'elem_area': elem_area}


def get_geom_limits(mesh: np.ndarray):
    '''
    Computes the geometric limits of the mechanical specimen.
    
    Parameters:
    -----------
        mesh : numpy.ndarray
            Array of nodal coordinates

    Returns:
    --------
        x_min : float
            Minimum x coordinate
        x_max : float
            Maximum x coordinate
        y_min : float
            Minimum y coordinate
        y_max : float
            Maximum y coordinate
    '''

    x_min = min(mesh[:,1])
    x_max = max(mesh[:,1])
    y_min = min(mesh[:,-1])
    y_max = max(mesh[:,-1])

    return x_min, x_max, y_min, y_max


def get_b_bar(bc_settings: dict, b_glob: np.ndarray, global_dofs: list):
    '''
    Computes a modified strain-displacement matrix, given a set of pre-defined boundary conditions.

    Parameters:
    -----------
        bc_settings : dict
            dictionary of boundary conditions
        b_glob : numpy.ndarray
            global strain-displacement matrix
        global_dofs : list
            list of global degrees of freedom

    Returns:
    --------
        b_bar : numpy.ndarray
            modified strain-displacement matrix
        active_dof : list
            list of active/free degrees of freedom
    '''

    b_bar = copy.deepcopy(b_glob)

    # Degrees of freedom to apply boundary conditions
    bc_fixed = []
    bc_slaves = []
    bc_masters = []

    for _, props in bc_settings['b_conds'].items():

        edge_dof_x = list(props['dof'][::2]-1)
        edge_dof_y = list(props['dof'][1::2]-1)
            
        master_dof = list(props['m_dof']-1)
        slave_dof = list(set(list(props['dof']-1)) - set(master_dof))

        # Set bc along x-direction
        if props['cond'][0] == 0:
            pass
        elif props['cond'][0] == 1:
            bc_fixed += edge_dof_x
        elif props['cond'][0] == 2:
            b_bar[:, master_dof[0]] += np.sum(b_bar[:,slave_dof[::2]],1)
            bc_slaves += slave_dof[::2]
            bc_masters.append(master_dof[0])
        
        # Set bc along y-direction
        if props['cond'][1] == 0:
            pass
        elif props['cond'][1] == 1:
            bc_fixed += edge_dof_y
        elif props['cond'][1] == 2:
            b_bar[:, master_dof[1]] += np.sum(b_bar[:,slave_dof[1::2]],1)
            bc_slaves += slave_dof[1::2]
            bc_masters.append(master_dof[1])

    # Checking for incompatible boundary conditions
    if len(list(set(bc_masters).intersection(bc_fixed)))!=0:
        raise Exception('Incompatible BCs, adjacent boundary conditions cannot be both fixed/uniform').with_traceback()
        
    # Defining the active degrees of freedom
    active_dof = list(set(global_dofs)-set(sum([bc_fixed,bc_slaves],[])))   

    # Discarding redundant boundary conditions
    b_bar = b_bar[:,active_dof]
    
    return b_bar, active_dof

def get_b_inv(mat: np.ndarray):
    '''
    Computes the pseudoinverse of a matrix.

    Makes use of Pytorch to perform the operation using the GPU, for faster computation time.

    Parameters:
    -----------
        mat : numpy.ndarray
            matrix to invert
    '''

    # Converting to pytorch and sending tensor to GPU
    mat_ = torch.from_numpy(mat).float().cuda()
    t1_start = time.process_time()
    # Computing pseudo-inverse matrix
    mat_inv = torch.linalg.pinv(mat_)
    t1_stop = time.process_time()
    print("Elapsed time for matrix inversion:", t1_stop-t1_start)

    return mat_inv

def get_glob_strain_disp(elements: list, total_dof: int, bc_settings: dict):
    '''
    Computes the global strain-displacemente matrix.

    Parameters:
    -----------
        elements : list
            list of Element objects
        total_dof : int
            total number of degrees of freedom
        bc_settings : dict
            dictionary containing the boundary settings for a mechanical specimen
    
    Returns:
    --------
        b_glob : np.ndarray
            global strain-displacemente matrix
    '''

    N_PTS = len(elements)  # Number of elements
    N_COMPS = 3  # Number of strain components

    # Initializing global strain-displacement matrix (B)
    b_glob = np.zeros([N_COMPS * N_PTS, total_dof]) 

    # Assembly of global strain-displacement matrix (B)
    for i, element in enumerate(elements):
        
        b_glob[N_COMPS*i:N_COMPS*i + N_COMPS, element.global_dof-1] += element.b_el()

    return b_glob

    # # Initializing modified strain-displacement matrix (B_)
    # b_bar = copy.deepcopy(b_glob)

    # # Degrees of freedom to apply boundary conditions
    # bc_fixed = []
    # bc_slaves = []
    # bc_masters = []

    # for edge, props in bc_settings['b_conds'].items():

    #     edge_dof_x = list(props['dof'][::2]-1)
    #     edge_dof_y = list(props['dof'][1::2]-1)
            
    #     master_dof = list(props['m_dof']-1)
    #     slave_dof = list(set(list(props['dof']-1)) - set(master_dof))

    #     # Set bc along x-direction
    #     if props['cond'][0] == 0:
    #         pass
    #     elif props['cond'][0] == 1:
    #         bc_fixed += edge_dof_x
    #     elif props['cond'][0] == 2:
    #         b_bar[:, master_dof[0]] += torch.sum(b_bar[:,slave_dof[::2]],1)
    #         bc_slaves += slave_dof[::2]
    #         bc_masters.append(master_dof[0])
        
    #     # Set bc along y-direction
    #     if props['cond'][1] == 0:
    #         pass
    #     elif props['cond'][1] == 1:
    #         bc_fixed += edge_dof_y
    #     elif props['cond'][1] == 2:
    #         b_bar[:, master_dof[1]] += torch.sum(b_bar[:,slave_dof[1::2]],1)
    #         bc_slaves += slave_dof[1::2]
    #         bc_masters.append(master_dof[1])
        
    # # Defining the active degrees of freedom
    # actDOFs = list(set(G_DOF)-set(sum([bc_fixed,bc_slaves],[])))
    
    # # Checking for incompatible boundary conditions
    # if len(list(set(bc_masters).intersection(bc_fixed)))!=0:
    #     raise Exception('Incompatible BCs, adjacent boundary conditions cannot be both fixed/uniform').with_traceback()

    # # Discarding redundant boundary conditions
    # b_bar = b_bar[:,active_dofs].cuda()
    # print('Calculating B_bar matrix')
    # t1_start = time.process_time() 
    # # Computing pseudo-inverse strain-displacement matrix
    # b_inv = torch.linalg.pinv(b_bar)
    # t1_stop = time.process_time()
    # print("Elapsed time for matrix inversion:", t1_stop-t1_start)
    
    # return b_glob, b_inv, actDOFs
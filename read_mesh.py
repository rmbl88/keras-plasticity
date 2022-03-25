import os
from matplotlib.pyplot import connect
import pandas as pd
import sys
from io import StringIO
import numpy as np

def get_substring_index(list, sub):

    return next((s for s in list if sub in s), None)

def get_bc_dofs(nodes_list):
    return np.array(sum([[2*i-1,2*i] for i in nodes_list],[]))

inp_file = ''
for r, d, f in os.walk('./'):
        for file in f:
            if '.inp' in file:
                inp_file = file

lines = []
with open(inp_file) as f:
    lines = f.readlines()
f.close()

start_part = lines.index(get_substring_index(lines,'*Part, name'))
end_part = lines.index(get_substring_index(lines,'*End Part'))

lines = lines[start_part:end_part+1]

start_mesh = lines.index(get_substring_index(lines,'*Node'))
end_mesh = lines.index(get_substring_index(lines,'*Element, type'))

start_connect = end_mesh
end_connect = lines.index(get_substring_index(lines,'*Nset, nset'))

mesh = ''.join(lines[start_mesh+1:end_mesh]).replace(' ','').split('\n')
mesh = pd.read_csv(StringIO('\n'.join(mesh)),names=['node','x','y'])

connect_str = ''.join(lines[start_connect+1:end_connect]).replace(' ','').split('\n')[:-1]

elem_nodes = len(connect_str[0].split(','))-1

connectivity = pd.read_csv(StringIO('\n'.join(connect_str)),names=['id']+['n%i'% i for i in range(elem_nodes)])
dof = [[int(j) for j in i.split(',')][1:] for i in connect_str]
dof = np.array([sum([[2*i-1,2*i] for i in a],[]) for a in dof])

print('hey')

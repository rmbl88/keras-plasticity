import os
from matplotlib.pyplot import connect
import pandas as pd
import sys
if sys.version_info[0] < 3: 
    from StringIO import StringIO
else:
    from io import StringIO

def get_element_substring(list, sub):

    return next((s for s in list if sub in s), None)

inp_file = ''
for r, d, f in os.walk('./'):
        for file in f:
            if '.inp' in file:
                inp_file = file

lines = []
with open(inp_file) as f:
    lines = f.readlines()
f.close()

start_part = lines.index(get_element_substring(lines,'*Part, name'))
end_part = lines.index(get_element_substring(lines,'*End Part'))

lines = lines[start_part:end_part+1]

start_mesh = lines.index(get_element_substring(lines,'*Node'))
end_mesh = lines.index(get_element_substring(lines,'*Element, type'))

start_connect = end_mesh
end_connect = lines.index(get_element_substring(lines,'*Nset, nset'))

mesh = ''.join(lines[start_mesh+1:end_mesh]).replace(' ','').split('\n')
mesh = pd.read_csv(StringIO('\n'.join(mesh)),names=['node','x','y'])

connectivity = ''.join(lines[start_connect+1:end_connect]).replace(' ','').split('\n')[:-1]
connectivity = [[int(j) for j in i.split(',')][1:] for i in connectivity]
connectivity = [sum([[2*i-1,2*i] for i in a],[]) for a in connectivity]

#
#n_elems = len(connectivity[0].split(','))-1

#3connectivity = pd.read_csv(StringIO('\n'.join(connectivity)),names=['id']+['node%i'%(i) for i in range(n_elems)])

print('hey')

from asyncore import read
from cProfile import label
from mimetypes import init
import os
import matplotlib.pyplot as plt
import pandas as pd
import sys
from io import StringIO
import numpy as np
from torch import meshgrid
from constants import TRAIN_MULTI_DIR
from matplotlib.gridspec import GridSpec

from functions import load_dataframes       

# Loading data
df_list, _ = load_dataframes(TRAIN_MULTI_DIR)

# Merging training data
data = pd.concat(df_list, axis=0, ignore_index=True)

#trial = data[data['tag']=='m80_b80_x']
d = data[(data['id']==9) & (data['tag']=='m80_b80_x')][['exx_t','eyy_t','exy_t','sxx_t','syy_t','sxy_t']]

# Calculate correlation between each pair of variable
corr_matrix=d.corr()
 
# Can be great to plot only a half matrix
# Generate a mask for the upper triangle
mask = np.zeros_like(corr_matrix)
mask[np.triu_indices_from(mask)] = True
#mask=mask[1:, :-1]
#corr_matrix=corr_matrix.iloc[1:,:-1].copy()
import seaborn as sns
# Draw the heatmap with the mask
sns.heatmap(corr_matrix, mask=mask, square=True,annot=True, annot_kws={"size": 7},linewidths=5,vmin=-1,vmax=1,cmap=sns.diverging_palette(0, 230, 90, 60, as_cmap=True))
plt.show()

elements = list(set(data['id']))

eps = dict.fromkeys(elements)
eps_past = dict.fromkeys(elements)
s = dict.fromkeys(elements)
de = dict.fromkeys(elements)

for elem in elements:

    s[elem] = trial[trial['id']==elem][['sxx_t','syy_t','sxy_t']].values
    eps[elem] = trial[trial['id']==elem][['exx_t','eyy_t','exy_t']].values
    eps_past[elem] = trial[trial['id']==elem][['exx_t-1dt','eyy_t-1dt','exy_t-1dt']].values
    #de[elem] = trial[trial['id']==elem][['exx_dt', 'eyy_dt', 'exy_dt']].values

fig = plt.figure()
fig.suptitle('%s' % ('m80_b80_x'))
fig.set_size_inches(16, 4, forward=True)
fig.subplots_adjust(wspace=0.6,bottom=0.25)
fig.tight_layout()
g_s = GridSpec(1, 3, figure=fig)

ax1 = fig.add_subplot(g_s[0,0])
ax2 = fig.add_subplot(g_s[0,1])
ax3 = fig.add_subplot(g_s[0,2])

for k,v in eps.items():
    e = v
    e_past = eps_past[k]
    stress = s[k]

    de = e-e_past

    ax1.plot(de[:,0],stress[:,0],label='Elem #%i' % (k))
    ax1.set_ylabel(r'$\sigma_{xx}$')
    ax2.plot(de[:,1],stress[:,1],label='Elem #%i' % (k))
    ax2.set_ylabel(r'$\sigma_{yy}$')
    ax3.plot(de[:,2],stress[:,2],label='Elem #%i' % (k))
    ax3.set_ylabel(r'$\sigma_{xy}$')

    for ax in fig.axes:
        ax.set_xlabel(r'$\Delta\varepsilon=\varepsilon^{t}-\varepsilon^{t-1}$')


handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center',ncol=len(elements))
# plt.legend(loc='best')
plt.show()

print('hey')

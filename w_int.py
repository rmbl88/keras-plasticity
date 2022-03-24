from cProfile import label
import pandas as pd
import joblib

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

w_int = joblib.load('w_int.pkl')
w_int_real = joblib.load('w_int_real.pkl')
w_ext = joblib.load('w_ext.pkl')

vfs = w_int.columns.to_list()[1:]

epochs= list(set(w_int['epoch']))

fig, axes = plt.subplots(1,3)
fig.set_size_inches(19.2,10.8,forward=True)
fig.suptitle('Internal Virtual Work\n\nm=10 b=270 | 1000 time points | 80/20 train/test split | On test data')

k=0
axes = np.reshape(axes,(1,3))
for i in range(axes.shape[0]):
    for j in range(axes.shape[1]):
        vf = vfs[k]
        ax=axes[i][j]
        ax.set_title('%s' % vf)
        
        wint = w_int[vf]
        wint_real = w_int_real[vf]
        wext = w_ext[vf]
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Wint [J]')
        ax.plot(epochs,wint,label='w_pred')
        ax.plot(epochs,wint_real,label='w_real')
        ax.plot(epochs,wext,label='w_ext')
        anchored_text = AnchoredText(('W_int_real=%.4e\nW_ext=%.4e\nW_pred=%.4e')%(wint_real.iloc[-1],wext.iloc[-1],wint.iloc[-1]), loc=2)
        ax.add_artist(anchored_text)
        k +=1

plt.legend(loc='best')
# plt.show()
plt.savefig('wint.png', dpi=100, bbox_inches='tight', format='png')

# k = 0
# for i in range(axes.shape[0]):
#     for j in range(axes.shape[1]):
#         elem = elem_ids[k]
#         ax = axes[i][j]
#         wint = w_int[w_int['id']==elem]
#         wint_real = w_int_real[w_int['id']==elem]
#         ax.set_title('Elem: %i' % elem)
#         for vf in vfs:
#             #delta = abs(w_int[vf]-w_int_real[vf])/w_int_real[vf]
#             ax.plot(w_int['epoch'], w_int[vf], label= vf + '- w_pred')
#             ax.plot(w_int['epoch'], w_int_real[vf], label= vf + '- w_real')
#         k += 1

print('hey')
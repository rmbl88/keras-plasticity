import joblib
import os
import glob

from plot_utils import plot_rafr

SPECIMENS = ['d_shaped', 's_shaped', 'sigma_shaped']

files = glob.glob('rafr_*.pkl')

for specimen in SPECIMENS:
    
    rafr_objs = dict.fromkeys([f.split('_')[-1][:-4] for f in files if specimen in f])
    for i, (model, _) in enumerate(rafr_objs.items()):
        rafr_objs[model] = [joblib.load(f) for f in files if (specimen in f) and (model in f)][0]

    plot_rafr(rafr_objs, specimen, 'plots')

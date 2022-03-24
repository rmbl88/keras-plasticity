import enum
from turtle import width
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functions import load_dataframes, data_sampling
from constants import TRAIN_MULTI_DIR, DATA_SAMPLES
import seaborn as sns

# Loading data
df_list, _ = load_dataframes(TRAIN_MULTI_DIR)

# Sampling data pass random seed for random sampling
sampled_dfs = data_sampling(df_list, DATA_SAMPLES)

# Merging training data
data = pd.concat(sampled_dfs, axis=0, ignore_index=True)

elem_ids = list(set(data['id'].values))

heights = dict.fromkeys(elem_ids)
bins = dict.fromkeys(elem_ids)
vars = ['id','exx_t','eyy_t','exy_t','sxx_t','syy_t','sxy_t']

df = data[vars]
c_map = sns.color_palette("Paired", 9)
for i, var in enumerate(vars[1:]):
    
    d = df[['id', var]]
    sns.histplot(data=d, x=var, hue='id', multiple="stack", bins=20, palette=c_map).set_title(var)
    
    plt.show()

               



    
    
    


        
    
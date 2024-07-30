import json
import shutil
import joblib
import os
import glob
import munch
import torch
import yaml

def load_pkl(file: str):

    return joblib.load(file)

def scan_ann_files(run: str, dir: str, key: str):

    SCAN_DIR = os.path.join('outputs', dir, 'models', run)
    
    for f in glob.glob(os.path.join(SCAN_DIR, f'*{key}*')):
    
        if key in f:
            file = f           

    return file 

def load_file(run: str, dir: str, key: str):
    
    f = scan_ann_files(run, dir, key)

    return load_pkl(f)

def get_ann_model(run: str, dir: str):
    
    f = scan_ann_files(run, dir, '.pt')
    
    return torch.load(f)

def load_config(path: str, config_name=None):

    with open(os.path.join(path, config_name) if config_name != None else path, 'r') as f:

        data = yaml.safe_load(f)

    return munch.munchify(data)

def save_config(new_config, out_file: str):

    with open(out_file, 'w') as file:

        yaml.dump(munch.unmunchify(new_config), 
                  file, 
                  default_flow_style=False, 
                  sort_keys=False)
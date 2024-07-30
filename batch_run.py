import os
import requests
import time
from time import gmtime
from time import strftime
import glob
import sys
from subprocess import *
from tqdm import tqdm
from datetime import datetime
from constants import FORMAT_PBAR
from dotenv import load_dotenv

load_dotenv()

def telegram_send(payload=None, url=None, chat_id=None):
    
    requests.post(url, json={'chat_id': chat_id,
                              'parse_mode': 'HTML',
                              'text': payload})

SEND_URL = f"https://api.telegram.org/bot{os.environ['API_TOKEN']}/sendMessage"

CONFIG_ROOT_PATH = './config'

#MODEL_TYPE = 'direct'
MODEL_TYPE = 'vfm'

CONFIG_TYPE_PATH = os.path.join(CONFIG_ROOT_PATH, f'batch_configs_{MODEL_TYPE}')

SCRIPT_FILE = 'vfm_gru_model_training.py' if MODEL_TYPE == 'vfm' else 'sbvfm_gru_model_training_direct.py'


if __name__ == "__main__":

    config_files = glob.glob(os.path.join(CONFIG_TYPE_PATH,'*.yaml'))

    pbar = tqdm(config_files, bar_format=FORMAT_PBAR)

    for i, config in enumerate(pbar):
        
        if i == 0:
            now = datetime.now()
            message = f'<b>**Batch training session launched**</b>\n\n  &gt; Start date: {now.strftime("%d/%m/%Y @ %H:%M:%S")}\n  &gt; Configs to run: {len(config_files)}'
            telegram_send(message, SEND_URL, os.environ['CHAT_ID'])

        config_name = config.split("\\")[-1]
        pbar.set_description(f'Running configuration: {config_name}')
        
        save = sys.stdout
        sys.stdout = open(os.devnull, 'w')

        message = f'<b>**Status**</b>\n\n  &gt; Running configuration: {i + 1} of {len(config_files)}\n  &gt; File: {config_name}'
        telegram_send(message, SEND_URL, os.environ['CHAT_ID'])
        
        t_start = time.time()

        p = Popen(f'conda activate newPytorchGPU & python {SCRIPT_FILE} {config}', shell=True)
        
        p.wait()
        exit_code = p.returncode
        sys.stdout = save

        t_end = time.time()

        if exit_code == 0:
           status = 'SUCCESS'
        else:
            status = 'CRASHED'

        message = f'<b>**Status**</b>\n\n  &gt; File: {config_name}\n  &gt; Status: {status}\n  &gt; Elapsed: {strftime("%H:%M:%S", gmtime(t_end-t_start))}'

        telegram_send(message, SEND_URL, os.environ['CHAT_ID'])

        time.sleep(1)
        pbar.update(1)

now = datetime.now()
message = f'<b>**Batch training session finished**</b>\n\n  &gt; End date: {now.strftime("%d/%m/%Y @ %H:%M:%S")}'
telegram_send(message, SEND_URL, os.environ['CHAT_ID'])
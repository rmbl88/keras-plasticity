from functions import data_seletion, load_data

TRAIN_DIR = 'data/training/'
TRAIN_SAMPLES = 500

df_list = load_data(TRAIN_DIR)

df_train = data_seletion(df_list, TRAIN_SAMPLES)



# Defining model training constants
SEED = 444
VAL_DIR = 'data/validation/'
VAL_DIR_MULTI = 'data/validation_multi/9-elem/'
TRAIN_DIR = 'data/training/'
TRAIN_MULTI_DIR = 'data/training_multi/9-elem/'
DATA_SAMPLES = 500
TEST_SIZE = 0.33

ELEM_AREA = 1
ELEM_THICK = 0.5
LENGTH = 3.0

PRE_TRAINING = False

# Customizing matplotlib
PARAMS = {'text.usetex' : True,
          'text.latex.preamble' : r"\usepackage{amsmath}",
          'font.size' : 12,
          'font.family' : 'serif',
          'font.serif' : ['Palatino Linotype'],
          'lines.linewidth' : 0.75,
          'axes.spines.top' : False,
          'axes.spines.right' : False,
          'axes.labelpad' : 7.5,
          'lines.linewidth' : 1,
          'legend.loc' : 'best',
          'legend.frameon' : False,
          'patch.force_edgecolor' : True,
          'figure.max_open_warning': 0,
}
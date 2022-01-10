
# Defining model training constants
SEED = 444
VAL_DIR = 'data/validation/'
VAL_DIR_MULTI = 'data/validation_multi/9-elem-elastoplastic/'
TRAIN_DIR = 'data/training/'
TRAIN_MULTI_DIR = 'data/training_multi/9-elem-200-plastic/'
DATA_SAMPLES = 200
TEST_SIZE = 0.2

ELEM_AREA = 1
ELEM_THICK = 0.1
LENGTH = 3.0

PRE_TRAINING = False

# Customizing matplotlib
PARAMS = {'text.usetex' : True,
          'text.latex.preamble' : r"\usepackage{amsmath,newpxtext,newpxmath}",
          'font.size' : 10,
          'font.family' : 'serif',
          'font.serif' : ['Palatino Linotype'],
          'lines.linewidth' : 0.75,
          'axes.spines.top' : False,
          'axes.spines.right' : False,
          'axes.labelsize': 7,
          'axes.labelpad' : 6.5,
          'lines.linewidth' : 1,
          'legend.loc' : 'best',
          'legend.frameon' : False,
          'patch.force_edgecolor' : True,
          'figure.max_open_warning': 0,
          #'axes.spines.left': True,
          #'axes.spines.bottom': True,
          #'axes.spines.top':    True,
          #'axes.spines.right':  True,
          #'xtick.top': True,
          #'ytick.right': True,
          'xtick.direction' : 'in',
          'ytick.direction': 'in',
          'xtick.color': 'b0b0b0',
          'ytick.color': 'b0b0b0',
          'xtick.major.width': 0.1,
          'ytick.major.width': 0.1,
          'legend.fontsize': 7,
          'xtick.labelsize': 7,
          'ytick.labelsize': 7
          
}
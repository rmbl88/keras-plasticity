
# Defining model training constants
#SEED = 70988
SEED = 9567
VAL_DIR = 'data/validation/'
VAL_DIR_MULTI = 'data/validation_multi/crux-plastic/'
TRAIN_DIR = 'data/training/'
TRAIN_MULTI_DIR = 'data/training_multi/crux-plastic/'

# DATA_SAMPLES = 51
LOOK_BACK = 1
TEST_SIZE = 0.3


LENGTH = 3.0
ELEM_AREA = LENGTH**2 / 9
ELEM_THICK = 0.1

FORMAT_PBAR='{l_bar}{bar:50}{r_bar}{bar:-50b}'


PRE_TRAINING = False

# Customizing matplotlib
PARAMS = {'text.usetex' : True,
          'text.latex.preamble' : r"\usepackage{amsmath,newpxtext,newpxmath}",
          'font.size' : 10,
          'font.family' : 'serif',
          'font.serif' : ['Palatino Linotype'],
          'lines.linewidth' : 0.75,
          'axes.labelsize': 8,
          'axes.labelpad' : 6.5,
          'lines.linewidth' : 0.6,
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
          'xtick.color': '000000',
          'ytick.color': '000000',
          'xtick.major.width': 0.1,
          'ytick.major.width': 0.1,
          'legend.fontsize': 7,
          'xtick.labelsize': 7,
          'ytick.labelsize': 7
          
}

# Customizing matplotlib
PARAMS_CONTOUR = {
        'text.usetex' : True,
        'text.latex.preamble' : r"\usepackage{amsmath,newpxtext,newpxmath}",
        'font.size' : 9,
        'font.family' : 'serif',
        'font.serif' : ['Palatino Linotype'],
        'axes.labelpad' : 6.5,
        'figure.max_open_warning': 0,
        'xtick.direction' : 'out',
        'ytick.direction': 'out',
        'xtick.color': '000000',
        'ytick.color': '000000',  
}
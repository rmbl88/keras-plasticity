
# Defining model training constants
#SEED = 70988
SEED = 9567

VAL_DIR = 'data/validation/'
VAL_DIR_MULTI = 'data/validation_multi/crux-new/'
TRAIN_DIR = 'data/training/'
TRAIN_MULTI_DIR = 'data/training_multi/crux-plastic/'

# DATA_SAMPLES = 51
LOOK_BACK = 1
TEST_SIZE = 0.3


LENGTH = 3.0
ELEM_AREA = LENGTH**2 / 9
ELEM_THICK = 0.1

FORMAT_PBAR='{l_bar}{bar:50}{r_bar}{bar:-50b}'
FORMAT_PBAR_SUB='{l_bar}{bar:20}{r_bar}{bar:-20b}'

PRE_TRAINING = False

# Customizing matplotlib
PARAMS = {'text.usetex' : True,
          'text.latex.preamble' : r"\usepackage{amsmath,newpxtext,newpxmath,xfrac,bm}",
          'font.size' : 11,
          'font.family' : 'serif',
          'font.serif' : ['Palatino Linotype'],
          'lines.linewidth' : 0.85,
          'axes.labelsize': 7.5,
          'axes.labelpad' : 6,
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
          'xtick.major.width': 0.125,
          'ytick.major.width': 0.125,
          'xtick.major.size': 3,
          'ytick.major.size': 3,
          'legend.fontsize': 7.5,
          'xtick.labelsize': 7.5,
          'ytick.labelsize': 7.5
          
}

# Customizing matplotlib
PARAMS_CONTOUR = {
        'text.usetex' : True,
        'text.latex.preamble' : r"\usepackage{amsmath,newpxtext,newpxmath,xfrac,bm}",
        'font.size' : 14,
        'font.family' : 'serif',
        'font.serif' : ['Palatino Linotype'],
        'axes.labelpad' : 6.5,
        'figure.max_open_warning': 0,
        'xtick.direction' : 'out',
        'ytick.direction': 'out',
        'xtick.color': '#000000',
        'ytick.color': '#000000',
        'legend.fontsize': 17,
        'xtick.labelsize': 17,
        'ytick.labelsize': 17,
        'axes.labelsize': 17  
}
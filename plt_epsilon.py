import numpy as np
import pylab as plt

from constants import LENGTH

x = np.linspace(0,3,11)
y = np.linspace(0,3,11)

X, Y = np.meshgrid(x,y)

zeros = np.zeros_like(X)
ones = np.ones_like(Y)
LENGTH = 3.0

vfs = {
    1: [zeros, -(x**2-LENGTH)*y/LENGTH**2],
    2: [],
    3: [],
    4: []
}

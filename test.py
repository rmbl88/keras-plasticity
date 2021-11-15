import numpy as np
import matplotlib.pyplot as plt
from constants import *
from mpl_toolkits import mplot3d
import math

 # Defining nodal coordinates
coords = np.linspace(0, int(LENGTH), 20)

x, y = np.meshgrid(coords, coords)

Z = np.sin(x**2*math.pi/LENGTH)/LENGTH**2


fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(x, y, Z, rstride=1, cstride=1,cmap='viridis',edgecolor='black')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.show()
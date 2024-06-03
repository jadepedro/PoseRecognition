"""Simple standalone testing script for animation effect. Not used in the project"""

from numpy import linspace
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy import signal

# Creating 3D figure
fig = plt.figure(figsize=(8, 8))
ax = plt.axes(projection='3d')

# Creating Dataset
color_cycle = plt.rcParams['axes.prop_cycle']()


# 360 Degree view
for angle in range(0, 90):
    x = linspace(0, 1, 51)
    a = x*angle * (1 - x)
    ax.plot3D(x, a, **next(color_cycle))

    ax.view_init(angle, 90-angle)

    plt.draw()
    plt.pause(1)
    plt.cla()


#plt.show()
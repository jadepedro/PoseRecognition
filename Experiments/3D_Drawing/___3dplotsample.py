"""Simple standalone testing script for animation effect. Not used in the project"""

import matplotlib.pyplot as plt
import numpy as np




fig = plt.figure(figsize=(4,4))

ax = fig.add_subplot(111, projection='3d')

ax.scatter(2,3,4) # plot the point (2,3,4) on the figure

plt.show()

plt.ion()



for i in range(50):
    y = np.random.random([10,1])
    plt.plot(y)
    plt.draw()
    plt.pause(0.0001)
    plt.clf()
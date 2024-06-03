import numpy as np
import matplotlib.pyplot as plt
import time
from Graphics import BlitManager as bm

class GraphicsHelper:
    """
    A helper class for creating and updating a real-time animated plot.

    Parameters:
    - x_start (float): The starting value of the x-axis.
    - x_end (float): The ending value of the x-axis.
    - y_min (float, optional): The minimum value of the y-axis. Default is -1.
    - y_max (float, optional): The maximum value of the y-axis. Default is 1.
    """

    def __init__(self, x_start, x_end, y_min=-1, y_max=1):
        """
        Initializes the GraphicsHelper object.

        Args:
        - x_start (float): The starting value of the x-axis.
        - x_end (float): The ending value of the x-axis.
        - y_min (float, optional): The minimum value of the y-axis. Default is -1.
        - y_max (float, optional): The maximum value of the y-axis. Default is 1.
        """
        # Prepare X axis data
        self.x = np.linspace(x_start, x_end, 100)
        # Prepare Y axis data
        self.y = np.zeros(100)

        # make a new figure
        self.fig, self.ax = plt.subplots()
        # set the limits of the y-axis
        self.ax.set_ylim(y_min, y_max)

        (self.ln,) = self.ax.plot(self.x, self.y, animated=True)
        # add a text
        self.fr_number = self.ax.annotate(
            "0",
            (0, 1),
            xycoords="axes fraction",
            xytext=(10, -10),
            textcoords="offset points",
            ha="left",
            va="top",
            animated=True,
        )
        self.bm = bm.BlitManager(self.fig.canvas, [self.ln, self.fr_number])
        # make sure our window is on the screen and drawn
        plt.show(block=False)
        #plt.pause(1)
    
    def set_ydata(self, func):
        """
        Updates the y-axis data of the plot.

        Args:
        - func (function): A function that takes x-axis values as input and returns corresponding y-axis values.
        """
        # update the artists
        self.ln.set_ydata(func(self.x))
  
    def set_text(self, text):
        """
        Sets the text annotation in the plot.

        Args:
        - text (str): The text to be displayed.
        """
        self.fr_number.set_text(text)
    
    def update(self):
        """
        Updates the plot by blitting the changes.
        """
        # tell the blitting manager to do its thing
        self.bm.update()

    def add_y_and_shift(self, y):
        """
        Adds a new y-axis value to the plot and shifts the existing values.

        Args:
        - y (float): The new y-axis value to be added.
        """
        # shift the old values one position
        self.y[0:-1] = self.y[1:]
        # add a new value y at the topmost position of the y array
        self.y[-1] = y
        self.ln.set_ydata(self.y)


# Example usage:
if False:
    gh = GraphicsHelper(0, 2 * np.pi, -2.5, 2.5)
    for j in range(500):
        #gh.set_ydata(lambda x: np.abs(np.sin(x + (j / 100) * np.pi)))
        gh.add_y_and_shift(np.sin(j / 100 * np.pi))
        gh.set_text(f"frame: {j}")
        gh.update()
        time.sleep(0.01)
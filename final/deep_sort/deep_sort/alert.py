import matplotlib.pyplot as plt
import numpy as np

def show_direction(ax,theta,frame):

    #좌표 plot
    for thetha_single in theta:
        r = np.ones(np.shape(thetha_single))
        ax.quiver(thetha_single, r, np.pi, -1, color='black', angles="xy", scale_units='xy', scale=1.)
        ax.plot(thetha_single, r, color='black', marker='o', markersize=5)













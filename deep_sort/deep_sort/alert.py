import matplotlib.pyplot as plt
import numpy as np

def show_direction(ax,theta,frame):

    #좌표 plot
    for theta_single in theta:
        real_x = theta_single * 57.29578
        if (real_x > 0 and real_x < 50) or (real_x < 360 and real_x > 330):
            theta_single = 0
        r = np.ones(np.shape(theta_single))
        ax.quiver(theta_single, r, np.pi, -1, color='black', angles="xy", scale_units='xy', scale=1.)
        ax.plot(theta_single, r, color='black', marker='o', markersize=5)


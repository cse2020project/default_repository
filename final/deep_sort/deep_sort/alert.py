import matplotlib.pyplot as plt
import numpy as np

def show_direction(ax,theta,bbox_size,isCloser):

    #concat
    all=np.vstack((theta,bbox_size,isCloser))
    all=all.T
    #좌표 plot
    for single in all:
        if single[2]==False:
            continue
        real_x = single[0] * 57.29578
        if (real_x > 0 and real_x < 50) or (real_x < 360 and real_x > 330):
            theta_single = 0
        r = np.ones(np.shape(single[0]))
        ax.quiver(single[0], r, np.pi, -1, color='red', angles="xy", scale_units='xy', scale=1.,width=single[1]/1000*0.005)













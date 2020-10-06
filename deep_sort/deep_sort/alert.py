import matplotlib.pyplot as plt
import numpy as np

def show_direction(theta):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='polar') # 1,1,1그리드

    # Plot origin (agent's start point) - 원점=보행자
    ax.plot(0, 0, color='red', marker='o', markersize=20, alpha=0.3)
    # plot(a,b) a=라디안, b=지름0~1사이

    # ax.quiver(theta, r, u, v)
    r = np.ones(np.shape(theta))
    ax.quiver(theta, r, np.pi, -1, color='black', angles="xy", scale_units='xy', scale=1.)
    ax.plot(theta, r, color='black', marker='o', markersize=5)

    # Plot configuration
    ax.set_rticks([])
    ax.set_rmin(0)
    ax.set_rmax(1)
    ax.set_thetalim(0, 2*np.pi)
    ax.set_xticks(np.linspace(0, 2*np.pi, 8, endpoint=False))
    ax.grid(False)
    ax.set_theta_direction(-1) #시계방향 극좌표
    ax.set_theta_zero_location("S") #0도가 어디에 있는지-S=남쪽

    plt.show(block=False)
    plt.pause(0.5)
    
#if __name__ == '__main__':
#    show_direction(np.array([2*np.pi, 0.5*np.pi]))

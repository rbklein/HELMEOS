"""
    Contains plotting functions
"""

import matplotlib.pyplot as plt
from setup import h_topography

def plot_all(x, u):
    """
        Plots the water height and discharge
    """
    fig, ax = plt.subplots(1,3)
    ax[0].plot(x, h_topography, label = 'Bottom')
    ax[0].plot(x, u[0] + h_topography, label = 'Water level')
    ax[1].plot(x, u[1])
    ax[2].plot(x, u[1]/u[0])
    ax[0].set_xlabel(r'$x$')
    ax[0].set_ylabel(r'$h$')
    ax[1].set_xlabel(r'$x$')
    ax[1].set_ylabel(r'$hu$')
    ax[0].set_title('Water height')
    ax[1].set_title('Discharge')
    ax[2].set_title('Velocity')
    ax[0].grid()
    ax[1].grid()
    ax[2].grid()
    ax[0].legend()

def show():
    """
        Shows all generated figures on screen
    """
    plt.show()
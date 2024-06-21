"""
    Contains plotting functions
"""

import matplotlib.pyplot as plt

from setup_thermodynamics import *
from entropy import entropy_variables

def plot_conserved(x, u):
    """
        Plots the water height and discharge
    """
    fig, ax = plt.subplots(3,1)
    ax[0].plot(x, u[0])
    ax[1].plot(x, u[1])
    ax[2].plot(x, u[2])
    ax[0].set_xlabel(r'$x$')
    ax[0].set_ylabel(r'$\rho$')
    ax[1].set_xlabel(r'$x$')
    ax[1].set_ylabel(r'$m$')
    ax[2].set_xlabel(r'$x$')
    ax[2].set_ylabel(r'$E$')
    #ax[0].set_title('Density')
    #ax[1].set_title('Momentum')
    #ax[2].set_title('Total energy')
    ax[0].grid()
    ax[1].grid()
    ax[2].grid()

def plot_primitive(x, u):
    pass

def plot_thermodynamic(x, u):
    rho = u[0]
    T = thermodynamics.solve_temperature_from_conservative(u)
    p = pressure(rho, T)
    s = physical_entropy(rho, T)
    e = internal_energy(rho, T)
    
    fig, ax = plt.subplots(4,1)
    ax[0].plot(x, p)
    ax[1].plot(x, s)
    ax[2].plot(x, e)
    ax[3].plot(x, T)
    ax[0].set_xlabel(r'$x$')
    ax[0].set_ylabel(r'$p$')
    ax[1].set_xlabel(r'$x$')
    ax[1].set_ylabel(r'$s$')
    ax[2].set_xlabel(r'$x$')
    ax[2].set_ylabel(r'$e$')
    ax[3].set_xlabel(r'$x$')
    ax[3].set_ylabel(r'$T$')
    #ax[0].set_title('Pressure')
    #ax[1].set_title('Entropy')
    #ax[2].set_title('Internal \n energy')
    #ax[3].set_title('Temperature')
    ax[0].grid()
    ax[1].grid()
    ax[2].grid()
    ax[3].grid()

def plot_entropy(x, u):
    eta = entropy_variables(u)

    print(eta.shape)

    fig, ax = plt.subplots(3,1)
    ax[0].plot(x, eta[0])
    ax[1].plot(x, eta[1])
    ax[2].plot(x, eta[2])
    ax[0].set_xlabel(r'$x$')
    ax[0].set_ylabel(r'$\eta_1$')
    ax[1].set_xlabel(r'$x$')
    ax[1].set_ylabel(r'$\eta_2$')
    ax[2].set_xlabel(r'$x$')
    ax[2].set_ylabel(r'$\eta_3$')
    ax[0].grid()
    ax[1].grid()
    ax[2].grid()


def show():
    """
        Shows all generated figures on screen
    """
    plt.show()
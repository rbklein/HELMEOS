"""
    Contains plotting functions
"""

import matplotlib.pyplot as plt

from setup_thermodynamics import *
from entropy import entropy_variables

import Maxwell

def plot_pv(points = [], VLE = False):
    """
        Plot a p-v diagram for the reduced van der waals equations (nondimensionalized w.r.t. critical point)

        If points is a 2d-array it will be plotted as set of points in the p-v plane
    """
    num_points_plot = 10000

    T_normal = jnp.linspace(0.3, 1.7, 15)
    v_normal = jnp.linspace(1/3+1e-2, 100, num_points_plot)
    p_vT = jax.jit(lambda v, T: pressure(1/v * rho_c, T * T_c) / p_c)

    fig, ax = plt.subplots()

    count = 0
    for T in T_normal:
        lcolor = [0.5,0.5,0.5]
        lwidth = 0.5
        if count == 7:
            lcolor = 'r'
            lwidth = 2.0
        if count == 6:
            lcolor = 'b'
            lwidth = 2.0
        p = p_vT(v_normal, T * jnp.ones(num_points_plot))
        ax.semilogx(v_normal, p, linewidth=lwidth, color=lcolor)
        count += 1

    if points is not []:
        T = thermodynamics.solve_temperature_from_conservative(points)
        p_normal = pressure(points[0], T) / p_c
        v_normal = rho_c / points[0]
        ax.scatter(v_normal, p_normal, marker = '+')

    ax.semilogx(1,1,'ro', markersize = 10)

    ax.set_xlim(1./3+1e-2,20)
    ax.set_ylim(1e-3,2)
    ax.set_xlabel("reduced specific volume $v_r$")
    ax.set_ylabel("reduced pressure $p_r$")

    if VLE:
        psat, v1, v2, v, vss, pss = Maxwell.solve_VLE_pressure_in_interval(T_normal[6])
        print(psat, v1, v2)
        ax.semilogx([v1, v2], [psat, psat], 'g', linewidth=2)
        ax.semilogx(v, psat * jnp.ones(v.shape), marker = '+')
        ax.plot(vss, pss)


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



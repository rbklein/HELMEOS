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
    num_points_plot = 1000

    T_normal = jnp.linspace(0.6, 1.7, 14)
    v_normal = jnp.logspace(jnp.log10(1/3+1e-2), jnp.log10(100), num_points_plot)
    p_vT = jax.jit(lambda v, T: pressure(1/v * rho_c, T * T_c) / p_c)

    fig, ax = plt.subplots()

    for T in T_normal:
        lcolor = [0.5,0.5,0.5]
        lwidth = 0.5
        p = p_vT(v_normal, T * jnp.ones(num_points_plot))
        ax.semilogx(v_normal, p, linewidth=lwidth, color=lcolor)

    T = 1
    p = p_vT(v_normal, T * jnp.ones(num_points_plot))

    lcolor = 'tab:red'
    lwidth = 2.0
    ax.semilogx(v_normal, p, linewidth=lwidth, color=lcolor)
    ax.semilogx(1,1,'o', markersize = 10, color=lcolor)

    if points is not []:
        T = T_from_u(points)
        p_normal = pressure(points[0], T) / p_c
        v_normal = rho_c / points[0]
        ax.scatter(v_normal, p_normal, marker = '+', color = 'tab:orange')

    ax.set_xlim(1./3+1e-2,20)
    ax.set_ylim(0,4)
    ax.set_xlabel("reduced specific volume $v_r$")
    ax.set_ylabel("reduced pressure $p_r$")

    if VLE:
        plot_VLE(fig, ax)

def plot_VLE(fig, ax):
    """
        Add the VLE region to an existing p-v plot
    """
    T_plot = jnp.linspace(0.6, 0.99, 20)
    T_plot = jnp.flip(T_plot)

    p_VLE_arr   = [1.0]
    v1_arr      = [1.0]
    v2_arr      = [1.0]

    for T in T_plot:
        p_VLE, v1, v2 = Maxwell.solve_VLE_pressure_in_interval(T, 0.01, (1 / (b_VdW * rho_c)))

        p_VLE_arr.append(p_VLE)
        v1_arr.append(v1)
        v2_arr.append(v2)

    lcolor = 'tab:blue'
    lwidth = 1.5

    ax.plot(v1_arr, p_VLE_arr, '-+', linewidth=lwidth, color=lcolor)
    ax.plot(v2_arr, p_VLE_arr, '-+', linewidth=lwidth, color=lcolor)


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
    ax[0].grid()
    ax[1].grid()
    ax[2].grid()

def plot_primitive(x, u):
    pass

def plot_thermodynamic(x, u):
    rho = u[0]
    T = T_from_u(u)
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
    ax[0].grid()
    ax[1].grid()
    ax[2].grid()
    ax[3].grid()

def plot_entropy(x, u):
    eta = entropy_variables(u)

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



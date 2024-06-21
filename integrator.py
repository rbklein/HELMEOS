"""
    Contains numerical time integrators

    dudt is assumed as a function taking the current (intermediate) state u and a reference state u_ref so that:
    dudt(u,u_ref)
"""

from functools import partial

from config_discretization import *

import entropy
import discrete_gradient
import minimization

#implement cfl time step computation

@partial(jax.jit, static_argnums = 1)
def RK4(u, dudt):
    """
        The classical 4-th order Runge-Kutta time integrator
    """
    k1 = dudt(u)
    k2 = dudt(u + k1 * dt / 2)
    k3 = dudt(u + k2 * dt / 2)
    k4 = dudt(u + k3 * dt)
    return u + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

@partial(jax.jit, static_argnums = 1)
def TVDRK3(u, dudt):
    """
        The classical 3-rd order TVD Runge-Kutta time integrator by Gottlieb and Shu from "Total variation diminishing Runge-Kutta schemes"
    """
    u1 = u + dt * dudt(u)
    u2 = 3/4 * u + 1/4 * u1 + 1/4 * dt * dudt(u1)
    un =  1/3 * u + 2/3 * u2 + 2/3 * dt * dudt(u2)
    return un

@partial(jax.jit, static_argnums = 1)
def Tadmor_midpoint(u, dudt):
    """
        Locally entropy conservative Crank-Nicolsen time integrator by Tadmor with Gonzalez discrete gradient "Entropy stability theory for difference 
        approximations of nonlinear conservation laws and related time-dependent problems"
    """

    #minimization requires vector data but u is matrix
    u_guess         = jnp.reshape(u, 2 * num_cells)

    #evaluates entropy average between vectorized new state and the old state u
    average_u       = lambda u_new: entropy.conservative_variables(discrete_gradient.Gonzalez(jnp.reshape(u_new, (2,-1)), u))

    #evaluates FOM residual at Gonzalez entropy average
    residual_Crank_Nicolson = lambda u_new: jnp.reshape(jnp.reshape(u_new, (2,-1)) - u - dt * dudt(average_u(u_new)), (2 * num_cells))

    u_new = minimization.newton_raphson(residual_Crank_Nicolson, u_guess, 1e-6)

    #reshape new state back to matrix
    return jnp.reshape(u_new, (2,-1))

def return_integrator(which_integrator):
    """
        Returns the specific numerical time integration function that will be used for computations

        Options:
            - RK4
            - Modified Crank-Nicolson
    """
    match which_integrator:
        case "RK4":
            """
                The classical 4-th order Runge-Kutta time integrator
            """
            integrator = RK4
        case "TVDRK3":
            """
                The classical 3-rd order TVD Runge-Kutta time integrator
            """
            integrator = TVDRK3
        case "TADMOR":
            """
                The locally entropy conservative modified Crank-Nicolson method by Tadmor
            """
            integrator = Tadmor_midpoint

    return integrator

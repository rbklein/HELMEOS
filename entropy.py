"""
    Contains functions related to the PDE entropy of the shallow water equations

    name clash with thermodynamics entropy function
"""
from config_discretization import *
from setup_thermodynamics import *

@jax.jit
def PDE_entropy(u):
    """
        Computes the PDE entropy of the shallow water equations
    """
    T = thermodynamics.solve_temperature_from_conservative(u)
    s = entropy(u[0], T)
    return - u[0] * s

@jax.jit
def entropy_variables(u):
    """
        Computes the entropy variables from the conserved variables of the shallow water equations
    """
    T = thermodynamics.solve_temperature_from_conservative(u)
    s = entropy(u[0], T)
    p = pressure(u[0], T)
    
    eta1 = -s + 1 / (u[0] * T) * (u[2] - u[1]**2 / u[0] + p)
    eta2 = u[1] / (u[0] * T)
    eta3 = 1 / T

    return jnp.array([eta1, eta2, eta3], dtype=DTYPE)
    
@jax.jit
def conservative_variables(eta):
    """
        Computes the conservative variables from the entropy variables
    """
    pass

    #u1 = (2 * eta[0] + eta[1]**2) / (2 * g)
    #u2 = u1 * eta[1]
    #return jnp.array([u1, u2], dtype=DTYPE)
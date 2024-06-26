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
    T = T_from_u(u)
    s = physical_entropy(u[0], T)
    return - u[0] * s

'''
def generate_entropy_variables():
    Jac_PDE_entropy = jax.jacfwd(PDE_entropy)

    def _entropy_variables(u):
        return Jac_PDE_entropy(u[:,None])[0,:,0]
    
    entropy_variables = jax.jit(jax.vmap(_entropy_variables, (1), (1)))
    return entropy_variables

entropy_variables = generate_entropy_variables()
'''

@jax.jit
def entropy_variables(u):
    """
        Computes the entropy variables from the conserved variables of the shallow water equations
    """
    T = T_from_u(u)
    s = physical_entropy(u[0], T)
    p = pressure(u[0], T)
    
    eta1 = -s + 1 / (u[0] * T) * (u[2] - u[1]**2 / u[0] + p)
    eta2 = u[1] / (u[0] * T)
    eta3 = -1 / T

    return jnp.array([eta1, eta2, eta3], dtype=DTYPE)

@jax.jit
def entropy_flux_potential(u):
    """
        Computes the entropy flux potential for an arbitrary equation of state
    """
    T = T_from_u(u)
    p = pressure(u[0], T)
    return (u[1] * p) / (u[0] * T)

@jax.jit
def conservative_variables(eta):
    """
        Computes the conservative variables from the entropy variables
    """
    pass
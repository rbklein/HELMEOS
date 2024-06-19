"""
    Combines all element of the spatial discretization i.e.:
        - entropy conservative flux
        - entropy dissipation operator
        - entropy dissipation limiter
"""

from config_discretization import *
from setup import *

@jax.jit
def dudt(u):
    """
        Returns the ODEs corresponding to the semi-discrete shallow water equations using the specified 
        entropy conservative flux, entropy dissipation operator and limiter
    """
    F = f_cons(padder(u,pad_width_flux))
    D = f_diss(padder(u,pad_width_diss), lim)
    return - ((F[:,1:] - F[:,:-1]) + (D[:,1:] - D[:,:-1])) / dx 


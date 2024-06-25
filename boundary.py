"""
    Implements boundary conditions

    Boundary conditions (except for periodic) are typically not entropy stable therefore accumulation of entropy can take place at nontrivial boundaries
"""

from functools import partial

from config_discretization import *

pad_system = jax.vmap(jnp.pad, (0, None, None), 0)
pad_system.__doc__ = "pads axis 1 of 2D array by specified width and using specified mode"

@partial(jax.jit, static_argnums = 1)
def periodic_pad(u, pad_width):
    """
        Implements periodic ghost cells with a width given by pad_width
    """
    u_pad = pad_system(u, pad_width, "wrap")
    return u_pad

@partial(jax.jit, static_argnums = 1)
def transmissive_pad(u, pad_width):
    """
        Toro's transmissive condition eq. 14.6 in "Riemann Solvers and Numerical Methods for Fluid Dynamics"
    """
    u_pad = pad_system(u, pad_width, "symmetric")
    return u_pad

def return_padder(which_boundary):
    """
        Returns the specific solution value padding function that will be used for computations

        Options:
            - Periodic
            - Transmissive
    """
    match which_boundary:
        case "PERIODIC":
            """
                Pad periodically
            """
            padder = periodic_pad
        case "TRANSMISSIVE":
            """
                Pad with Toro's transmissive boundary conditions eq. 14.6 in "Riemann Solvers and Numerical Methods for Fluid Dynamics"
            """
            padder = transmissive_pad
    
    return padder

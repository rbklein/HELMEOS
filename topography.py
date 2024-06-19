"""
    Implements functions for bottom topographies

    Assumes non-time-dependent topographies
"""

from config_discretization import *

import flux

@jax.jit
def jump(quantity):
    """
        compute jump in scalar-valued quantity on grid
    """
    return quantity[1:] - quantity[:-1]

def flat_topography(x, params):
    """
        Return the flat topography
    """
    return x * 0.0

def thacker_topography(x, params):
    """
        Return the bottom topography of Thacker's test case see "Swashes: a compilation of shallow water analytic solutions for hydraulic and
        environmental studies"
    """
    h_0, a = params
    return h_0 * ((x/a)**2 - 1)

def rectangular_bump_topography(x, params):
    """
        Return a rectangular bump 
    """
    bump_height, bump_width = params
    return jnp.where(jnp.abs(x) < bump_width / 2, bump_height, 0)

def alessia_topography(x, params):
    """
        Return Alessia's non-zero topography

        note: can be parameterized
    """
    x_off = x + 750
    h = 0.0 * x
    h = h + jnp.where((x_off <= 562.5)*(x_off > 487.5), 4*jnp.exp(2 - 150 / (x_off - 487.5)), 0)
    h = h + jnp.where((x_off <= 637.5)*(x_off > 562.5), 8 - 4*jnp.exp(2 - 150 / (637.5 - x_off)), 0)
    h = h + jnp.where((x_off <= 862.5)*(x_off > 637.5), 8, 0)
    h = h + jnp.where((x_off <= 937.5)*(x_off > 862.5), 8 - 4*jnp.exp(2 - 150 / (x_off - 862.5)), 0)
    h = h + jnp.where((x_off <= 1012.5)*(x_off > 937.5), 4*jnp.exp(2 - 150 / (1012.5 - x_off)), 0)
    return h

def return_topography(which_topography):
    """
        Returns the specific topography vector that will be used for computations

        Options:
            - Flat
            - Thacker
            - Rectangular bump
    """
    match which_topography:
        case "FLAT":
            """
                A flat topography
            """
            topography = flat_topography
        case "THACKER":
            """
                Thacker's topography
            """
            topography = thacker_topography
        case "RECTANGULAR_BUMP":
            """
                A rectangular bump
            """
            topography = rectangular_bump_topography
        case "ALESSIA":
            """
                Alessia's non-zero topography
            """
            topography = alessia_topography

    return topography

@jax.jit
def Fjordholm_source(u, b):
    """
        Computes the source term using the entropy conservative scheme of Fjordholm, Mishra and Tadmor see "Well-balanced and 
        energy stable schemes for the shallow water equations with discontinuous topography"

        Assumes u, b are padded appropriately
    """
    if TOPOGRAPHY == "FLAT":
        return jnp.zeros((2,num_cells), dtype=DTYPE)
    
    b_jump          = jump(b)
    h_mean          = flux.a_mean(u[0])
    h_b_jump_mean   = flux.a_mean(h_mean * b_jump)

    S1 = jnp.zeros(num_cells, dtype=DTYPE)
    S2 = - g / dx * h_b_jump_mean

    return jnp.array([S1, S2], dtype=DTYPE)
    
def return_source(which_source):
    """
        Returns the specific source discretization functiuon that will be used for computations

        Options:
            - Fjordholm, Mishra, Tadmor
    """
    match which_source:
        case "FJORDHOLM":
            """
                The source of Fjordholm, Mishra and Tadmor
            """
            assert(pad_width_source == 1)
            source = Fjordholm_source

    return source





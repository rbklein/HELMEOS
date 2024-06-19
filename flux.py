"""
    Contains functions to compute entropy conservative numerical fluxes

    The numerical flux functions assume the solution u to be padded
"""

from config_discretization import *
from setup_thermodynamics import *

#shift value for jnp.roll
shift = -1

@jax.jit
def logmean(quantity):
    """
        Robust computation of the logarithmic mean from Ismail and Roe "Affordable, 
        entropy-consistent Euler flux functions II: Entropy production at shocks"
    """
    a = quantity[1:]
    b = quantity[:-1]

    d = a / b
    f = (d - 1) / (d + 1)
    u = f**2
    F = jnp.where(u < 0.001, 1 + u / 3 + u**2 / 5 + u**3 / 7, jnp.log(d) / 2 / f)
    return (a + b) / (2 * F)

@jax.jit
def a_mean(quantity):
    """
        Computes arithmetic mean of quantity 

        The mean is taken between as many cells as possible without exceeding array dimensions
    """

    return 0.5 * (quantity[1:] + quantity[:-1])

@jax.jit
def Ismail_Roe_flux(u):
    """
        Computes flux of Ismail and Roe from "Affordable, 
        entropy-consistent Euler flux functions II: Entropy production at shocks"

        NOTE: internal energy should be computed consistently throughout all code as difference total energy and kinetic energy or from EoS
    """
    T = thermodynamics.solve_temperature_from_conservative(u)

    #rho_e = u[2] - 0.5 * u[1]**2 / u[0] 
    #p = (gamma - 1) * rho_e

    p = pressure(u[0], T)

    z = jnp.ones((3, num_ghost_cells_flux))
    z = z.at[1].set(u[1] / u[0])
    z = z.at[2].set(p)
    z = jnp.sqrt(u[0]/p) * z

    z1m = a_mean(z[0]) 
    z2m = a_mean(z[1])
    z3m = a_mean(z[2])
    z1ln = logmean(z[0])
    z3ln = logmean(z[2])

    F1 = z2m * z3ln
    F2 = z3m / z1m + z2m / z1m * F1
    F3 = 0.5 * (z2m / z1m) * ((gamma + 1) / (gamma - 1) * (z3ln / z1ln) + F2)
    return jnp.array([F1,F2,F3], dtype=DTYPE)



def return_flux(which_flux):
    """
        Returns the specific flux function that will be used for computations

        Options:
            - Ismail and Roe
    """
    match which_flux:
        case "ISMAIL_ROE":
            """
                The flux by Ismail and Roe
            """
            assert(pad_width_flux == 1)
            flux = Ismail_Roe_flux

    return flux
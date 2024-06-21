"""
    Contains functions to compute entropy conservative numerical fluxes

    The numerical flux functions assume the solution u to be padded
"""

from config_discretization import *
from setup_thermodynamics import *

from computational import logmean, a_mean

import entropy


@jax.jit
def Ismail_Roe_flux(u):
    """
        Computes flux of Ismail and Roe from "Affordable, 
        entropy-consistent Euler flux functions II: Entropy production at shocks"

        Flux is entropy conservative for the ideal gas equation of state
    """
    T = thermodynamics.solve_temperature_from_conservative(u)
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

@jax.jit
def naive_flux(u):
    rho_mean = a_mean(u[0])
    m_mean = a_mean(u[1])
    T = thermodynamics.solve_temperature_from_conservative(u)
    p = pressure(u[0], T)
    p_mean = a_mean(p)
    E_mean = a_mean(u[2])
    u_mean = a_mean(u[1] / u[0])

    F1 = rho_mean * u_mean
    F2 = m_mean * u_mean + p_mean
    F3 = u_mean * (E_mean + p_mean)
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
        case "NAIVE":
            assert(pad_width_flux == 1)
            flux = naive_flux


    return flux

if __name__ == "__main__":
    print(entropy(2,2))
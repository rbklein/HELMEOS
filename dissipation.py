"""
    Contains function to compute entropy dissipation operators to use with entropy conservative fluxes for the shallow water equations

    The dissipation operators assume the solution u to be padded
"""
from functools import partial

from config_discretization import *
from setup_thermodynamics import *

from computational import jump_vec, mul, arith_mean

import entropy

@jax.jit
def minmod(delta1, delta2):
    """
        Computes minmod limiter values

        Suitable for vector-valued inputes
    """
    delta3 = jnp.where(jnp.abs(delta2) > 0, delta2, 1e-14)
    ratio_delta = delta1 / delta3
    phi = jnp.where(ratio_delta < 0, 0, ratio_delta)
    return jnp.where(phi < 1, phi, 1)

def return_limiter(which_limiter):
    """
        Returns the specific limiter function that will be used for computations of the entropy dissipation operator

        Options:
            - Minmod
    """
    match which_limiter:
        case "MINMOD":
            """
                The minmod limiter
            """
            limiter = minmod

    return limiter

@partial(jax.jit, static_argnums = 1)
def first_order(u, limiter):
    """
        Computes a first-order accuracte entropy dissipation operator

        To be implemented
    """
    pass

@partial(jax.jit, static_argnums = 1)
def Roe_dissipation(u, limiter):
    """
        Computes second-order Roe entropy dissipation operator from the paper of Fjordholm, Mishra and Tadmor from "Arbitrarily High-order 
        Accurate Entropy Stable Essentially Nonoscillatory Schemes for Systems of Conservation Laws" using limiter

        Assumes u is padded with boundary values

        Assumes ideal gas equation of state (for now)
    """

    #number of cell interfaces that are not on the boundary of the grid (including ghost cells)
    num_inner_cell_interfaces = num_ghost_cells_diss - 1

    vel = u[1] / u[0]
    vel_mean = arith_mean(vel)

    T = thermodynamics.solve_temperature_from_conservative(u)
    p               = pressure(u[0], T) 
    p_mean          = arith_mean(p)
    rho_mean        = arith_mean(u[0])
    speed_of_sound  = jnp.sqrt(gamma * p_mean / rho_mean)

    E_mean = arith_mean(u[2])
    H_mean = (E_mean + p_mean) / rho_mean

    eta         = entropy.entropy_variables(u)
    eta_jump    = jump_vec(eta)

    #compute eigenvector at mean state between all inner interfaces
    R = jnp.ones((3,3,num_inner_cell_interfaces))
    R = R.at[1,0,:].set(vel_mean - speed_of_sound); R = R.at[1,1,:].set(vel_mean); R = R.at[1,2,:].set(vel_mean + speed_of_sound)
    R = R.at[2,0,:].set(H_mean - vel_mean * speed_of_sound); R = R.at[2,1,:].set(0.5 * vel_mean**2); R = R.at[2,2,:].set(H_mean + vel_mean * speed_of_sound)

    #compute eigenvalue matrices at mean state between all inner interfaces
    D = jnp.zeros((3,3,num_inner_cell_interfaces))
    D = D.at[0,0,:].set(jnp.abs(speed_of_sound - vel_mean)); D = D.at[1,1,:].set(jnp.abs(vel_mean)); D = D.at[2,2,:].set(jnp.abs(speed_of_sound + vel_mean))

    #compute inner product of entropy variable jump and eigenvector basis
    delta = mul(R.transpose(1,0,2), eta_jump)

    delta_L = delta[:,:num_cells+1]
    delta_C = delta[:,1:num_cells+2]
    delta_R = delta[:,2:]

    limiter_mean = 0.5 * (limiter(delta_L, delta_C) + limiter(delta_R, delta_C))

    #compute scaling coefficient eigenvalues
    S = jnp.zeros((3,3,num_cells + 1))
    S = S.at[0,0,:].set(1 - limiter_mean[0,:]); S = S.at[1,1,:].set(1 - limiter_mean[1,:]); S = S.at[2,2,:].set(1 - limiter_mean[2,:])

    return -mul(R[:,:,1:num_cells+2], mul(D[:,:,1:num_cells+2], mul(S, delta[:,1:num_cells+2]))) * 1/2


def return_dissipation(which_dissipation):
    """
        Returns the specific dissipation operator that will be used for computations of entropy stable numerical fluxes

        Options:
            - Roe dissipation
            - No dissipation
    """
    match which_dissipation:
        case "TECNO_ROE":
            """
                Roe-based dissipation operator from the TeCNO class schemes
            """
            assert(pad_width_diss == 2)
            dissipation = Roe_dissipation
        case "NONE":
            """
                No dissipation
            """
            assert(pad_width_diss == 1)
            dissipation = lambda u, lim: 0.0 * u

    return dissipation

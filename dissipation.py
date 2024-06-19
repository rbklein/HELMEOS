"""
    Contains function to compute entropy dissipation operators to use with entropy conservative fluxes for the shallow water equations

    The dissipation operators assume the solution u to be padded
"""
from functools import partial

from config_discretization import *

import entropy
import flux
import boundary
import discrete_gradient

@jax.jit
def jump_vec(quantity):
    """
        Compute jump in vector-valued quantity on grid 

        Computes as many jumps as possible given the quantity

        indices:
            quantity: 0 row index, 1 grid index
    """
    return quantity[:,1:] - quantity[:,:-1]

@jax.jit
def mul(A,v):
    """
        Computes matrix vector product at each node of the grid

        numpy matmul docs: 'If either argument is N-Dimensional, N > 2, it is treated as a stack of matrices residing in the last two indexes and broadcast accordingly.'

        indices:
            A: 0 row index, 1 column index, 2 grid index
            v: 0 vector component, 1 grid index + (a dummy index to enable use of jnp.matmul)
    """
    return jnp.matmul(A.transpose((2,0,1)), v[:,:,None].transpose((1,0,2)))[:,:,0].transpose()

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

    #theta = discrete_gradient.zero_by_zero(delta1, delta2)
    #phi = jnp.where(theta < 0., 0., jnp.where(theta > 1, 1, theta))
    #return phi

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

import matplotlib.pyplot as plt

@partial(jax.jit, static_argnums = 1)
def Roe_dissipation(u, limiter):
    """
        Computes second-order Roe entropy dissipation operator from the paper of Fjordholm, Mishra and Tadmor from "Arbitrarily High-order 
        Accurate Entropy Stable Essentially Nonoscillatory Schemes for Systems of Conservation Laws" using limiter

        Assumes u is padded with boundary values
    """

    #number of cell interfaces that are not on the boundary of the grid (including ghost cells)
    num_inner_cell_interfaces = num_ghost_cells_diss - 1

    h_mean      = flux.a_mean(u[0])
    hu_mean     = flux.a_mean(u[1])
    vel_mean    = hu_mean / h_mean
    vel_char    = jnp.sqrt(g * h_mean)

    eta         = entropy.entropy_variables(u)
    eta_jump    = jump_vec(eta)

    #compute eigenvector at mean state between all inner interfaces
    R = jnp.ones((2,2,num_inner_cell_interfaces))
    R = R.at[1,0,:].set(vel_mean - vel_char); R = R.at[1,1,:].set(vel_mean + vel_char)

    #compute eigenvalue matrices at mean state between all inner interfaces
    D = jnp.zeros((2,2,num_inner_cell_interfaces))
    D = D.at[0,0,:].set(jnp.abs(vel_mean - vel_char)); D = D.at[1,1,:].set(jnp.abs(vel_mean + vel_char))

    #compute inner product of entropy variable jump and eigenvector basis
    delta = mul(R.transpose(1,0,2), eta_jump)

    delta_L = delta[:,:num_cells+1]
    delta_C = delta[:,1:num_cells+2]
    delta_R = delta[:,2:]

    limiter_mean = 0.5 * (limiter(delta_L, delta_C) + limiter(delta_R, delta_C))

    #compute scaling coefficient eigenvalues
    S = jnp.zeros((2,2,num_cells + 1))
    S = S.at[0,0,:].set(1 - limiter_mean[0,:]); S = S.at[1,1,:].set(1 - limiter_mean[1,:])

    return -mul(R[:,:,1:num_cells+2], mul(D[:,:,1:num_cells+2], mul(S, delta[:,1:num_cells+2]))) * 1/2

def return_dissipation(which_dissipation):
    """
        Returns the specific dissipation operator that will be used for computations of entropy stable numerical fluxes

        Options:
            - Roe dissipation
            - Laplacian (not implemented)
            - No dissipation
    """
    match which_dissipation:
        case "TECNO_ROE":
            """
                Roe-based dissipation operator from the TeCNO class schemes
            """
            assert(pad_width_diss == 2)
            dissipation = Roe_dissipation
        case "LAPLACIAN":
            """
                Simple Laplacian dissipation
            """
            raise NotImplementedError("Laplacian dissipation has not been implemented yet")
        case "NONE":
            """
                No dissipation
            """
            assert(pad_width_diss == 1)
            dissipation = lambda u, lim: 0.0 * u

    return dissipation

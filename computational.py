"""
    Contains functions for generic and frequently recurring mathematical operations
"""

import numpy as np
from scipy.spatial import ConvexHull

from config_discretization import *

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
def jump(quantity):
    """
        Compute jump in scalar-valued quantity on grid 

        Computes as many jumps as possible given the quantity

        indices:
            quantity: 0 row index, 1 grid index
    """
    return quantity[1:] - quantity[:-1]

@jax.jit
def norm2_nodal(u):
    """
        Computes the squared norm of a vector in every mesh point
    """
    return jnp.sum(u**2, axis = 0)

@jax.jit
def norm2(u):
    """
        Computes the squared norm of a vector on the whole grid
    """
    return jnp.sum(norm2_nodal(u))

@jax.jit
def inner_nodal(u,v):
    """
        Computes the inner product of two vectors in every mesh point
    """
    return jnp.sum(u*v, axis = 0)

@jax.jit
def inner(u,v):
    """
        Computes the inner product between two vector on the whole grid
    """
    return jnp.sum(inner_nodal(u,v))

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
def log_mean(quantity):
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
def arith_mean(quantity):
    """
        Computes arithmetic mean of quantity 

        The mean is taken between as many cells as possible without exceeding array dimensions
    """

    return 0.5 * (quantity[1:] + quantity[:-1])

@jax.jit
def zero_by_zero(num, den):
    """
        Robust division operator for 0/0 = 0 scenarios (thanks to Alessia)
    """
    return den * (jnp.sqrt(2) * num) / (jnp.sqrt(den**4 + jnp.maximum(den, 1e-14)**4))

def convex_envelope(x, fs):
    """
        Compute indices of the lower convex envelope of a function, code adapted from:

            "https://gist.github.com/parsiad/56a68c96bd3d300cb92f0c03c68198ee"
    """
    N = fs.shape[0]
    
    fs_pad = np.empty(N+2)
    fs_pad[1:-1], fs_pad[0], fs_pad[-1] = fs, np.max(fs) + 1.0, np.max(fs) + 1.0
    
    x_pad = np.empty(N+2)
    x_pad[1:-1], x_pad[0], x_pad[-1] = x, x[0], x[-1]
    
    epi = np.column_stack((x_pad, fs_pad))
    hull = ConvexHull(epi)
    result = [v-1 for v in hull.vertices if 0 < v <= N]
    result.sort()
    
    return jnp.array(result, dtype=jnp.int32)
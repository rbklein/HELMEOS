"""
    Contains functions for generic and frequently recurring mathematical operations
"""

import numpy as np
import numpy.polynomial.legendre as leg

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




'''
def find_real_roots_in_interval(a,b,f):
    """
        Finds roots of a scalar-valued, univariate function f in interval [a,b] by fitting a polynomial of arbitrary degree (max 100)
    """
    x = np.linspace(a, b, 1000)
    y = f(x)

    import matplotlib.pyplot as plt

    plt.figure()

    plt.semilogx(x,y)
    plt.show()

    precision_threshold = 1e-12 
    degree              = 1
    max_degree          = 100  
    fitting_error       = np.inf

    while fitting_error > precision_threshold and degree <= max_degree:
        coefficients    = leg.legfit(x, y, degree)
        y_fit           = leg.legval(x, coefficients)
        fitting_error   = np.linalg.norm(y - y_fit)
        
        #print(f"Degree: {degree}, Fitting error: {fitting_error}")

        degree += 1

        roots                   = leg.legroots(coefficients)
        real_roots              = roots[np.isreal(roots)].real
        real_roots_in_interval  = real_roots[(real_roots >= a) & (real_roots <= b)]

    return real_roots_in_interval
'''

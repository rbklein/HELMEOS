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
def zero_by_zero(num, den):
    """
        Robust division operator for 0/0 = 0 scenarios (thanks to Alessia)
    """
    return den * (jnp.sqrt(2) * num) / (jnp.sqrt(den**4 + jnp.maximum(den, 1e-14)**4))

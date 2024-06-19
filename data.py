"""
    Function to handle data
"""

from config_discretization import *

def prepare_data():
    """
        return data array of size consistent with sample_rate and num_steps
    """
    arr = jnp.arange(0, num_steps+1, sample_rate)
    data = jnp.zeros((2, num_cells, len(arr)))
    return data

#@jax.jit
def gather_data(data, u, index):
    """
        Gather data in predefined data array at index
    """
    if index % sample_rate == 0:
        data = data.at[:,:,index//sample_rate].set(u)
    return data

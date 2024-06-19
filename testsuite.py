"""
    A testsuite containing initial conditions for different test cases
"""

from config_discretization import *

def return_case(test_case):
    """
        Returns initial conditions for test case given by test_case
    """
    match test_case:
        case "ACOUSTIC":
            initial_rho = lambda x, params: 1 + 0.2 * jnp.sin(2 * jnp.pi * x)
            initial_v = lambda x, params: 1.5 + 0.2 * 1.4 * jnp.sin(2 * jnp.pi * x)
            initial_p = lambda x, params: 1 + 0.2 * 1.4**2 * jnp.sin(2 * jnp.pi * x)
        case "CHAN":
            initial_rho = lambda x, params: 2 + 0.5 * jnp.exp(-100 * x**2)      
            initial_v = lambda x, params: 1/10 * jnp.exp(-100 * x**2)           
            initial_p = lambda x, params: initial_rho(x, params)**gamma  

    return initial_rho, initial_v, initial_p
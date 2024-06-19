"""
    A testsuite containing initial conditions for different test cases
"""

from config_discretization import *

def return_case(test_case):
    """
        Returns initial conditions for test case given by test_case
    """
    match test_case:
        case "DAM_BREAK":
            initial_h = lambda x, h, params = (): jnp.where(jnp.abs(x) < 0.2, 1.5, 1) - HEIGHT_IS_FREE_SURFACE * h(x,topography_params)   
            initial_hu = lambda x, h, params = (): x * 0.0 
        case "DROP":
            initial_h = lambda x, h, params = (): 0.1 * jnp.exp(-100 * x**2) + 1.0 - HEIGHT_IS_FREE_SURFACE * h(x,topography_params)  
            initial_hu = lambda x, h, params = (): x * 0 
        case "RIEMANN":
            initial_h = lambda x, h, params: jnp.where(x < 0, params[0], params[1]) - HEIGHT_IS_FREE_SURFACE * h(x,topography_params)  
            initial_hu = lambda x, h, params: jnp.where(x < 0, params[2], params[3]) * initial_h(x, h, params)
        case "ENTROPY_RIEMANN":
            initial_h = lambda x, h, params = (): jnp.where(jnp.abs(x) < 12.5, 1.5, 0.02) - HEIGHT_IS_FREE_SURFACE * h(x,topography_params)  
            initial_hu = lambda x, h, params = (): x * 0.0 
        case "POSITIVITY_RIEMANN":
            initial_h = lambda x, h, params = (): 5 + 0 * x - HEIGHT_IS_FREE_SURFACE * h(x,topography_params)  
            initial_hu = lambda x, h, params = (): jnp.where(x < 0.0, -35.0, 35.0) * initial_h(x, h, params)

    return initial_h, initial_hu
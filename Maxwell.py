"""
    Contains function to perform the Maxwell construction to discover the VLE of an arbitrary equation of state
"""

from config_discretization import *
from setup_thermodynamics import *

from computational import convex_envelope

def solve_VLE_pressure_in_interval(T, rho1 = 0.1, rho2 = 1):
    """
        Compute the VLE interval for a given isotherm by computing the lower convex envelope of the helmholtz energy as a function of 
        specific volume 'v' and determining for what v it differs from the helmholtz energy given by the equation of state, see 
        Markus Deserno "Van der Waals equation, Maxwell construction, and Legendre transforms"

        values are reduced

    """
    v           = jnp.logspace(jnp.log10(1/rho2), jnp.log10(1/rho1), 1000)
    helmholtz   = eos(1/v * rho_c, T * T_c) 

    envelope_indices = convex_envelope(v, helmholtz)

    index_v1 = jnp.nonzero((envelope_indices[1:] - envelope_indices[:-1]) != 1)[0][0]

    v2 = v[envelope_indices[index_v1+1]]
    v1 = v[index_v1]

    p_VLE = pressure((1/v2 * rho_c)[None], (T * T_c)[None])[0] / p_c 

    return p_VLE, v1, v2



    
'''
Slower code doing the same thing


from jax.scipy.integrate import trapezoid

def find_sign_change_indices(arr):
    signs = jnp.sign(arr)
    signs_L = signs[:-1]
    signs_R = signs[1:]
    sign_change_indices = jnp.nonzero((signs_L * signs_R) < 0)[0]
    return sign_change_indices

def solve_VLE_pressure_in_interval(T, rho1 = 0.1, rho2 = 10, p_guess = 1.0):
    """
        Find the saturation pressure for a reduced equation of state with a method
    """
    p   = p_guess
    dp  = 0.001

    iter    = 0
    maxiter = 1000
    area1   = 1.0
    area2   = 0.0

    num_nodes = 1000

    v1 = 1 / rho2
    v2 = 1 / rho1

    v = jnp.logspace(jnp.log10(v1), jnp.log10(v2), num_nodes)
    #v  = jnp.linspace(v1,v2,num_nodes)

    @jax.jit
    def pressure_residual(v, T, p):
        T_arr = T * jnp.ones(num_nodes)
        p_arr = p * jnp.ones(num_nodes)
        return pressure(1/v * rho_c, T_arr * T_c) / p_c - p_arr

    while iter <= maxiter:
        residual        = pressure_residual(v, T, p)
        root_indices    = find_sign_change_indices(residual)
        v_roots         = 0.5 * (v[root_indices] + v[root_indices + 1])

        if root_indices.shape[0] > 1:  
            area1 = trapezoid(residual[root_indices[0]:root_indices[1]+1], v[root_indices[0]:root_indices[1]+1])
            area2 = trapezoid(residual[root_indices[1]:root_indices[2]+1], v[root_indices[1]:root_indices[2]+1])
            area1 = jnp.abs(area1)
            area2 = jnp.abs(area2)

            if jnp.abs(area1 - area2) < 1e-8:
                break

            if area1 > area2:
                p = p - dp
                while p < 0:
                    p = p + dp
                    dp = dp / 10
                    p = p - dp
            else:
                p = p + dp
                dp = dp / 10
                p = p - dp
        else:
            p = p - dp
            while p < 0:
                p = p + dp
                dp = dp / 10
                p = p - dp

        iter += 1

        print(p, v_roots, area1, area2)

    return p, v[root_indices[0]], v[root_indices[2]]

'''




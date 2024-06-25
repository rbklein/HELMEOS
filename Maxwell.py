"""
    Contains function to perform the Maxwell construction to discover the VLE of an arbitrary equation of state
"""

from config_discretization import *
from setup_thermodynamics import *

#from computational import find_real_roots_in_interval

from jax.scipy.integrate import trapezoid

import matplotlib.pyplot as plt

def find_sign_change_indices(arr):
    signs = jnp.sign(arr)
    signs_L = signs[:-1]
    signs_R = signs[1:]
    sign_change_indices = jnp.nonzero((signs_L * signs_R) < 0)[0]
    return sign_change_indices

def solve_VLE_pressure_in_interval(T, rho1 = 0.1, rho2 = 10):
    """
        Find the saturation pressure for a reduced equation of state with a method
    """
    p   = 1.0
    dp  = 0.001

    iter    = 0
    maxiter = 1000
    area1   = 1.0
    area2   = 0.0

    num_nodes = 10000

    v1 = 1 / rho2
    v2 = 1 / rho1
    v  = jnp.linspace(v1,v2,num_nodes)

    @jax.jit
    def pressure_residual(v, T, p):
        T_arr = T * jnp.ones(num_nodes)
        p_arr = p * jnp.ones(num_nodes)
        return pressure(1/v * rho_c, T_arr * T_c) / p_c - p_arr

    vss = []
    pss = []

    while (jnp.abs(area1 - area2) > 1e-8) and (iter <= maxiter):
        residual        = pressure_residual(v, T, p)
        root_indices    = find_sign_change_indices(residual)
        v_roots         = 0.5 * (v[root_indices] + v[root_indices + 1])

        if root_indices.shape[0] == 1:
            vss.append(v_roots[0])
            pss.append(p)

        print(p, v_roots, jnp.abs(area1 - area2))

        if root_indices.shape[0] > 1:        
            area1 = trapezoid(residual[:root_indices[1]], v[:root_indices[1]])
            area2 = trapezoid(residual[root_indices[1]:], v[root_indices[1]:])
            area1 = jnp.abs(area1)
            area2 = jnp.abs(area2)

            if area1 > area2:
                p = p - dp
                if p < 0:
                    p = p + dp
                    dp = dp / 10
                    p = p - dp
            else:
                p = p + dp
                dp = dp / 10
                p = p - dp
        else:
            p = p - dp
            if p < 0:
                p = p + dp
                dp = dp / 10
                p = p - dp

        iter += 1

    return p, v[root_indices[0]], v[root_indices[2]], v, vss, pss



if __name__ == "__main__":
    arr = jnp.linspace(0, 10, 11) + 5
    print(arr)

    inds = jnp.nonzero(arr > 9)[0]
    print(inds)

    print(inds.shape[0])





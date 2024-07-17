"""
    Contains discrete gradient functions
"""

from config_discretization import *

from functools import partial
from computational import zero_by_zero, arith_mean_vec, jump_vec, jump, inner_nodal

@partial(jax.jit, static_argnums = (1,2))
def Gonzalez(u, f, df, diags = 1):
    """
        Computes the Gonzalez discrete gradient of grid function f with inputs u 
        appearing in paper: "Time integration and discrete Hamiltonian systems"

        Discrete gradient is computed on as many cell interfaces as possible, df is a function containing the gradient
        Allows for diagonal norm given by diags
    """
    u_mean = arith_mean_vec(u)
    u_diff = jump_vec(u)

    f_val   = f(u)
    f_diff  = jump(f_val)
    df_mean = df(u_mean)

    u_diff_scaled = diags * u_diff

    factor_num  = f_diff - inner_nodal(u_diff, df_mean)
    factor_den  = inner_nodal(u_diff_scaled, u_diff)
    factor      = zero_by_zero(factor_num, factor_den)

    disc_grad   = df_mean + factor[None, :] * u_diff_scaled 
    return disc_grad



'''
import entropy

@jax.jit
def Gonzalez(u1,u2):
    """
        Computes the Gonzalez discrete gradient appearing in paper: "Time integration and discrete Hamiltonian systems"
    """
    u_mean = 0.5 * (u1 + u2)
    u_jump = u1 - u2
    s_jump = entropy.PDE_entropy(u1) - entropy.PDE_entropy(u2)

    #computes squared norm at each point in grid
    u_norm_squared = jnp.sum(u_jump**2, axis=0)

    #entropy variables at arithmetic average of states
    eta_mean    = entropy.entropy_variables(u_mean)

    #Gonzalez correction factor
    factor_num      = (s_jump - jnp.sum(eta_mean * u_jump, axis = 0)) 
    factor_den      = u_norm_squared
    factor = zero_by_zero(factor_num, factor_den)
    
    #use broadcasting to multiply every component of u_jump by factor
    eta_gonzalez = eta_mean + factor[None,:] * u_jump

    return eta_gonzalez
'''
    
if __name__ == "__main__":
    @jax.jit
    def f(u):
        return u[0] + 2*u[1]
    
    def _f(u):
        f_val = f(u)
        return jnp.reshape(f_val, ())

    _df = jax.grad(_f)

    df = jax.jit(jax.vmap(_df, 1, 1))

    u1 = jnp.linspace(1,10,10)
    u2 = jnp.linspace(4,13,10)
    u = jnp.vstack((u1,u2))

    u_pad = jnp.pad(u, ((0,0), (20,13)), 'edge')

    diags = jnp.ones((2, 9))

    print(df(u))
    #print(df(u_pad))
    print(Gonzalez(u, f, df, diags))


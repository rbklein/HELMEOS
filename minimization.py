"""
    Contains minimization functions

    Used for implicit time integration
"""

from functools import partial

from config_discretization import *

@partial(jax.jit, static_argnums = 0)
def newton_raphson(f, x0, tol=1e-6, maxiter=20):
    """
        JIT-compileable implementation of the Newton-Raphson root finding algorithm (thanks in part to chatGPT (: )
    """

    # Define a JIT compiled function to compute the Jacobian of f
    jacobian_f = jax.jit(jax.jacfwd(f))

    # Initial guess
    x = jnp.array(x0)

    def cond_fun(val):
        x, fx, iter_count = val
        return jnp.logical_and(jnp.linalg.norm(fx) > tol, iter_count < maxiter)

    def body_fun(val):
        x, _, iter_count = val
        fx = f(x)
        Jx = jacobian_f(x)

        delta_x = jnp.linalg.solve(Jx, -fx)

        x = x + delta_x
        return x, f(x), iter_count + 1

    # Initial values for the loop
    init_val = (x, f(x), 0)

    # Use lax.while_loop for iteration
    x, fx, iter_count = jax.lax.while_loop(cond_fun, body_fun, init_val)

    return x


@partial(jax.jit, static_argnums = 0)
def levenberg_marquardt(f, x0, tol=1e-6, maxiter=100, lambda_init=1e-3):
    """
        JIT-compileable implementation of the Levenberg-Marquardt root finding algorithm (thanks in part to chatGPT (: )
    """

    # Define a JIT compiled function to compute the Jacobian of f
    jacobian_f = jax.jit(jax.jacfwd(f))

    # Initial guess and damping parameter
    x = jnp.array(x0)
    lambd = lambda_init

    def cond_fun(val):
        x, fx, _, iter_count = val
        return jnp.logical_and(jnp.linalg.norm(fx) > tol, iter_count < maxiter)

    def body_fun(val):
        x, fx, lambd, iter_count = val
        Jx = jacobian_f(x)
        A = Jx.T @ Jx
        g = Jx.T @ fx
        
        # Levenberg-Marquardt update
        delta_x = jnp.linalg.solve(A + lambd * jnp.eye(A.shape[0]), -g)
        x_new = x + delta_x
        fx_new = f(x_new)
        
        # Update lambda based on the improvement
        rho = (jnp.linalg.norm(fx) - jnp.linalg.norm(fx_new)) / (0.5 * delta_x.T @ (lambd * delta_x - g))
        lambd_new = jnp.where(rho > 0, lambd * jnp.maximum(1 / 3, 1 - (2 * rho - 1) ** 3), lambd * 2)
        
        # Accept or reject the new step
        x = jnp.where(rho > 0, x_new, x)
        fx = jnp.where(rho > 0, fx_new, fx)
        
        return x, fx, lambd_new, iter_count + 1

    # Initial values for the loop
    init_val = (x, f(x), lambd, 0)

    # Use lax.while_loop for iteration
    x, fx, lambd, iter_count = jax.lax.while_loop(cond_fun, body_fun, init_val)

    return x


def verbose_newton_raphson(f, x0, tol=1e-6, maxiter=20):
    """
        A simple Newton-Raphson implementation that gives iteration information (thank you chatGPT)
    """

    import matplotlib.pyplot as plt

    jacobian_f = jax.jacfwd(f)
    u = jnp.array(x0)

    for i in range(maxiter):
        fx = f(u)
        Jx = jacobian_f(u)
        
        #plt.figure()
        #plt.plot(u)

        #Jx_p = jnp.where(jnp.abs(Jx) > 5, 5, Jx)

        #plt.figure()
        #plt.imshow(Jx_p)

        #plt.show()

        if jnp.linalg.norm(fx) < tol:
            print(f'Converged in {i+1} iterations.')
            return u
        
        delta_x = jnp.linalg.solve(Jx, -fx)

        print(jnp.max(delta_x))

        u = u + delta_x

    print('Warning: Maximum number of iterations reached without convergence.')
    exit()
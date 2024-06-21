"""
    Contains functions to compute thermodynamic quantities
"""

from config_discretization import *

@jax.jit
def ideal_gas(rho, T):
    """
        Specific Helmholtz energy of the ideal gas equation of state
    """
    #f = T * (molecular_dofs/2 * gas_constant - molar_entropy_ref) - molecular_dofs/2 * gas_constant * T * jnp.log(T / T_ref) + gas_constant * T * jnp.log(rho / rho_ref)
    #return f / molar_mass
    return T / (gamma - 1) * (1 - jnp.log(T / (rho**(gamma - 1))))

@jax.jit
def Van_der_Waals(rho,T):
    """
        Specific Helmholtz energy of the Van der Waals equation of state

        still using gamma - 1 = 2 / num_dofs, which only holds for ideal gas
    """
    return T / (2 / molecular_dofs) * (1 - jnp.log(((1 - b_VdW * rho)**(2 / molecular_dofs) * T) / (rho**(2 / molecular_dofs)))) - rho * a_VdW

def return_eos(which_eos):
    """
        Return the specific Helmholtz free energy function given by which_eos
    """
    match which_eos:
        case "IDEAL":
            """
                The ideal gas equation of state in terms of its specific Helmholtz free energy
            """
            eos = ideal_gas
        case "VAN_DER_WAALS":
            """
                The Van der Waals equations of state in terms of its specific Helmholtz free energy
            """
            eos = Van_der_Waals
    return eos

def generate_pressure(eos):
    """
        Generates a function for thermodynamic pressure from the specific Helmholtz free energy function
    """

    #define eos function with output shape () for scalar inputs, required for grad
    def _eos(rho,T):
        f = eos(rho,T)
        return jnp.reshape(f, ())

    #vmap grad of _eos to support array inputs i.e. mesh values
    dAdrho = jax.vmap(jax.grad(_eos, argnums = 0), (0,0), 0)

    @jax.jit
    def pressure(rho, T):
        return rho**2 * dAdrho(rho, T) 
    
    return pressure

def generate_entropy(eos):
    """
        Generates a function for specific entropy from the specific Helmholtz free energy function
    """
    def _eos(rho,T):
        f = eos(rho,T)
        return jnp.reshape(f, ())

    dAdT = jax.vmap(jax.grad(_eos, argnums = 1), (0,0), 0)

    @jax.jit
    def entropy(rho, T):
        return -dAdT(rho, T)
    
    return entropy

def generate_internal_energy(eos):
    """
        Generates a function for specific internal energy from the specific Helmholtz free energy function
    """
    def _eos(rho,T):
        f = eos(rho,T)
        return jnp.reshape(f, ())

    dAdT = jax.vmap(jax.grad(_eos, argnums = 1), (0,0), 0)

    @jax.jit
    def internal_energy(rho, T):
        return eos(rho, T) - T * dAdT(rho, T)
    
    return internal_energy

def solve_temperature_from_pressure(rho, p):
    """
        Solve temperature profile from density and pressure profiles in a least squares manner
    """
    #for testing this is taking as ideal
    #T = p * molar_mass / (rho * gas_constant)
    #T = p / rho
    T = (p + a_VdW * rho**2) * (1 / rho - b_VdW) 
    return T

def solve_temperature_from_conservative(u):
    """
        Solve temperature profile from conservative variable profiles in a least squares manner
    """
    
    #T = (2 * molar_mass) / (molecular_dofs * gas_constant) * (u[2] / u[0] - 0.5 * u[1]**2 / u[0]**2)
    #T = (gamma - 1) * (u[2] / u[0] - 0.5 * u[1]**2 / u[0]**2)
    T = ((u[2] / u[0] - 0.5 * u[1]**2 / u[0]**2) + a_VdW * u[0]) / (molecular_dofs / 2)
    return T

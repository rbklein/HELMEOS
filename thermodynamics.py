"""
    Contains functions to compute thermodynamic quantities
"""

from config_discretization import *

@jax.jit
def ideal_gas(rho, T):
    """
        Specific Helmholtz energy of the ideal gas equation of state
    """
    return T / (gamma - 1) * (1 - jnp.log(T / (rho**(gamma - 1))))

@jax.jit
def Van_der_Waals(rho,T):
    """
        Specific Helmholtz energy of the Van der Waals equation of state

        critical point:
            - T_c   = 8/27 * a/b
            - rho_c = 1/3 * 1/b
            - p_c   = 1/27 * a/b^2 
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

def set_critical_points(which_eos):
    """
        Returns the critical points for a chosen Helmholtz free energy
    """
    match which_eos:
        case "IDEAL":
            """
                Ideal gas law: has no critical point thus set ones
            """
            rho_c = 1.0
            T_c = 1.0
            p_c = 1.0
        case "VAN_DER_WAALS":
            """
                Van der waals critical point
            """
            rho_c = 1/3 * 1/b_VdW
            T_c   = 8/27 * a_VdW/b_VdW
            p_c   = 1/27 * a_VdW/b_VdW**2 
    
    return rho_c, T_c, p_c

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

@jax.jit
def T_from_p_ideal(rho, p):
    """
        Solve temperature profile from density and pressure for ideal gas
    """
    T = p / rho
    return T

@jax.jit
def T_from_p_Van_der_Waals(rho, p):
    """
        Solve temperature profile from density and pressure for Van der Waals gas
    """
    T = (p + a_VdW * rho**2) * (1 / rho - b_VdW) 
    return T

def set_T_from_p(which_eos):
    """
        Return a function to compute temperature from the density and pressure using some equation of state
    """
    match which_eos:
        case "IDEAL":
            """
                The ideal gas equation of state
            """
            T_from_p = T_from_p_ideal
        case "VAN_DER_WAALS":
            """
                The Van der Waals equation of state
            """
            T_from_p = T_from_p_Van_der_Waals

    return T_from_p

@jax.jit
def T_from_u_ideal(u):
    """
        Solve temperature profile from conservative variable profiles for ideal gas
    """
    T = (gamma - 1) * (u[2] / u[0] - 0.5 * u[1]**2 / u[0]**2)
    return T

@jax.jit
def T_from_u_Van_der_Waals(u):
    """
        Solve temperature profile from conservative variable profiles for Van der Waals gas
    """
    T = ((u[2] / u[0] - 0.5 * u[1]**2 / u[0]**2) + a_VdW * u[0]) / (molecular_dofs / 2)
    return T


def set_T_from_u(which_eos):
    """
        Return a function to compute temperature from the conservative variables using some equation of state
    """
    match which_eos:
        case "IDEAL":
            """
                The ideal gas equation of state
            """
            T_from_u = T_from_u_ideal
        case "VAN_DER_WAALS":
            """
                The Van der Waals equation of state
            """
            T_from_u = T_from_u_Van_der_Waals

    return T_from_u




'''
def T_from_p(rho, p):
    """
        Solve temperature profile from density and pressure profiles
    """
    #for testing this is taking as ideal
    #T = p * molar_mass / (rho * gas_constant)
    #T = p / rho
    T = (p + a_VdW * rho**2) * (1 / rho - b_VdW) 
    return T
'''

'''
def T_from_u(u):
    """
        Solve temperature profile from conservative variable profiles
    """
    #T = (2 * molar_mass) / (molecular_dofs * gas_constant) * (u[2] / u[0] - 0.5 * u[1]**2 / u[0]**2)
    #T = (gamma - 1) * (u[2] / u[0] - 0.5 * u[1]**2 / u[0]**2)
    T = ((u[2] / u[0] - 0.5 * u[1]**2 / u[0]**2) + a_VdW * u[0]) / (molecular_dofs / 2)
    return T
'''

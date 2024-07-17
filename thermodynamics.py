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

def generate_gibbs_energy(eos):
    """
        Generates a function for specific gibbs free energy from the specific Helmholtz free energy function
    """
    def _eos(rho,T):
        f = eos(rho,T)
        return jnp.reshape(f, ())
    
    dAdrho = jax.vmap(jax.grad(_eos, argnums = 0), (0,0), 0)

    @jax.jit
    def gibbs_energy(rho, T):
        return eos(rho, T) + rho * dAdrho(rho, T) 
    
    return gibbs_energy

def generate_enthalpy(eos):
    """
        Generates a function for specific enthalpy from the specific Helmholtz free energy function
    """
    def _eos(rho,T):
        f = eos(rho,T)
        return jnp.reshape(f, ())
    
    dAdrho  = jax.vmap(jax.grad(_eos, argnums = 0), (0,0), 0)
    dAdT    = jax.vmap(jax.grad(_eos, argnums = 1), (0,0), 0)

    @jax.jit
    def enthalpy(rho, T):
        return eos(rho, T) + rho * dAdrho(rho, T) - T * dAdT(rho, T)
    
    return enthalpy

def generate_Chandrashekar_pressure_derivatives(eos):
    """
        Generates 
            d_{rho} (p/T) = 2 rho / T d_{rho} A + rho^2 / T d2_{rho,rho} A
        and 
            d_{1/T} (p/T) = rho^2 d_{rho} A - rho^2 T d2_{rho,T} A
        as required for the generalized Chandrashekar flux
    """
    def _eos(rho,T):
        f = eos(rho,T)
        return jnp.reshape(f, ())
    
    _dAdrho     = jax.grad(_eos, argnums = 0)
    _d2Adrho2   = jax.grad(_dAdrho, argnums = 0)
    _d2AdrhodT  = jax.grad(_dAdrho, argnums = 1)

    dAdrho      = jax.vmap(_dAdrho, (0,0), 0)
    d2Adrho2    = jax.vmap(_d2Adrho2, (0,0), 0)
    d2AdrhodT   = jax.vmap(_d2AdrhodT, (0,0), 0)

    @jax.jit 
    def dpTdrho(rho,T):
        return 2 * rho / T * dAdrho(rho, T) + rho**2 / T * d2Adrho2(rho, T)
    
    @jax.jit
    def dpTdbeta(rho,T):
        return rho**2 * dAdrho(rho, T) - rho**2 * T * d2AdrhodT(rho, T)
    
    """
        Input is required in the form u where u[0] = rho, u[1] = beta = 1/T
    """
    @jax.jit
    def Chandrashekar_pressure_derivative(u):
        return jnp.vstack((dpTdrho(u[0],1/u[1]), dpTdbeta(u[0],1/u[1])))

    return Chandrashekar_pressure_derivative

def generate_Chandrashekar_Gibbs_derivative(eos):
    """
        Generates
            d_{rho} (g/T) = 2 / T d_{rho} A + rho / T d2_{rho,rho} A
        and
            d_{1/T} (g/T) = H - rho T d2_{rho,T} A
        as required for the generalized Chandrashekar flux, where H is the enthalpy
    """
    def _eos(rho,T):
        f = eos(rho,T)
        return jnp.reshape(f, ())
    
    _dAdrho     = jax.grad(_eos, argnums = 0)
    _dAdT       = jax.grad(_eos, argnums = 1)
    _d2Adrho2   = jax.grad(_dAdrho, argnums = 0)
    _d2AdrhodT  = jax.grad(_dAdrho, argnums = 1)

    dAdrho      = jax.vmap(_dAdrho, (0,0), 0)
    dAdT        = jax.vmap(_dAdT, (0,0), 0)
    d2Adrho2    = jax.vmap(_d2Adrho2, (0,0), 0)
    d2AdrhodT   = jax.vmap(_d2AdrhodT, (0,0), 0)

    @jax.jit 
    def dgTdrho(rho,T):
        return 2 / T * dAdrho(rho, T) + rho / T * d2Adrho2(rho, T)
    
    @jax.jit
    def dgTdbeta(rho,T):
        return eos(rho, T) + rho * dAdrho(rho, T) - T * dAdT(rho, T) - rho * T * d2AdrhodT(rho, T)
    
    """
        Input is required in the form u where u[0] = rho, u[1] = beta = 1/T
    """
    @jax.jit
    def Chandrashekar_Gibbs_derivative(u):
        return jnp.vstack((dgTdrho(u[0],1/u[1]), dgTdbeta(u[0],1/u[1])))

    return Chandrashekar_Gibbs_derivative


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


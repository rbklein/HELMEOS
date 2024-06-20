
"""
    Sets up functions that can be used in the FOM module. 
    Compute initial conditions and specify functions based on configurations

    variables:
        - rho:  density
        - v:    velocity
        - p:    thermodynamic pressure
"""

from config_discretization import *
from setup_thermodynamics import *

import testsuite
import flux
import dissipation
import integrator
import boundary

initial_rho, initial_v, initial_p = testsuite.return_case(TEST_CASE)

f_cons      = flux.return_flux(FLUX)
f_diss      = dissipation.return_dissipation(DISSIPATION)
lim         = dissipation.return_limiter(LIMITER)
step        = integrator.return_integrator(INTEGRATOR)
padder      = boundary.return_padder(BOUNDARY)

rho_0   = initial_rho(x, initial_condition_params)
v_0     = initial_v(x, initial_condition_params)
p_0     = initial_p(x, initial_condition_params)

m_0     = rho_0 * v_0
T_0     = thermodynamics.solve_temperature_from_pressure(rho_0, p_0)
E_0     = rho_0 * internal_energy(rho_0, T_0) + 0.5 * rho_0 * v_0**2

u_0     = jnp.array([rho_0, m_0, E_0], dtype=DTYPE)


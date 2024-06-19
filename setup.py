
"""
    Compute initial conditions and specify functions based on configurations

    variables:
        - h:    water column height
        - hu:   discharge
"""

from config_discretization import *

import testsuite
import flux
import dissipation
import integrator
import boundary
import topography

initial_h, initial_hu = testsuite.return_case(TEST_CASE)

f_cons      = flux.return_flux(FLUX)
f_diss      = dissipation.return_dissipation(DISSIPATION)
f_source    = topography.return_source(SOURCE)
lim         = dissipation.return_limiter(LIMITER)
step        = integrator.return_integrator(INTEGRATOR)
padder      = boundary.return_padder(BOUNDARY)
topo        = topography.return_topography(TOPOGRAPHY)


b               = topo(padder(x[None,:], pad_width_source)[0,:], topography_params) 
h_topography    = topo(x, topography_params)

h_0     = initial_h(x, topo, initial_condition_params)
hu_0    = initial_hu(x, topo, initial_condition_params)

u_0     = jnp.array([h_0, hu_0], dtype=DTYPE)


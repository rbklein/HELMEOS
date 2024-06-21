"""
    Configure discretization options and parameters

    Discretization details:
        - Cell-centered finite volume method
        - Boundary conditions are implemented via ghost cells
        - Real-gas equations of state can be used
"""

import jax

#Datatypes
set_DTYPE = "DOUBLE"

match set_DTYPE:
    case "DOUBLE":
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp
        DTYPE = jnp.float64
    case "SINGLE":
        import jax.numpy as jnp
        DTYPE = jnp.float32


#Mesh [-length,length]
length      = 1
num_cells   = 1000
dx          = 2 * length / num_cells
x           = jnp.linspace(-length + 0.5 * dx, length - 0.5 * dx, num_cells, dtype = DTYPE)

#Temporal
time_final  = 0.3 #0.0001
num_steps   = 1000  #1
dt          = time_final / num_steps

#Boundary
#padding width necessary for implementing boundary conditions
pad_width_flux          = 1
pad_width_diss          = 2

num_ghost_cells_flux    = 2 * pad_width_flux + num_cells
num_ghost_cells_diss    = 2 * pad_width_diss + num_cells

#Initial conditions
initial_condition_params = ()

#thermodynamics (of CO2)
#gas_constant        = 8.314472
#molar_mass          = 28.96e-3              # Air: 28.96 g/mol      CO2: 44.01e-3 g/mol     
molecular_dofs      = 5                     # Air: 5                CO2: 5              

gamma               = 1.4  #should be equal to 1 + 2 / molecular_dofs according to wikipedia

#rho_ref             = 1.0
#T_ref               = rho_ref**(gamma-1) * molar_mass / gas_constant
#molar_entropy_ref   = 0.0

a_VdW = 1
b_VdW = 0.2

#Numerics
EOS         = "VAN_DER_WAALS"
TEST_CASE   = "CHAN"
FLUX        = "GONZALEZ"
LIMITER     = "MINMOD"
DISSIPATION = "TECNO_ROE"
INTEGRATOR  = "TVDRK3"
BOUNDARY    = "TRANSMISSIVE"

#data
sample_rate = 10
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
length      = 0.5
num_cells   = 2000
dx          = 2 * length / num_cells
x           = jnp.linspace(-length + 0.5 * dx, length - 0.5 * dx, num_cells, dtype = DTYPE)

#Temporal
time_final  = 0.3 #0.0001
num_steps   = 20000  #1
dt          = time_final / num_steps

#Boundary
#padding width necessary for implementing boundary conditions
pad_width_flux          = 1
pad_width_diss          = 2

num_ghost_cells_flux    = 2 * pad_width_flux + num_cells
num_ghost_cells_diss    = 2 * pad_width_diss + num_cells

#Initial conditions
initial_condition_params = ()

#Thermodynamics 
molecular_dofs      = 5    

gamma = 1.4  
a_VdW = 1.0
b_VdW = 0.2

#Numerics
EOS         = "VAN_DER_WAALS"
TEST_CASE   = "SOD"
FLUX        = "GENERALIZED_CHANDRASHEKAR" #"GONZALEZ"
LIMITER     = "MINMOD"
DISSIPATION = "TECNO_ROE"
INTEGRATOR  = "TVDRK3"
BOUNDARY    = "TRANSMISSIVE"   

#data
sample_rate = 10

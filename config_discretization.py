"""
    Configure discretization options and parameters

    Discretization details:
        - Cell-centered finite volume method
        - Boundary conditions are implemented via ghost cells
        - Non-flat bottom topography can be included

    Entropy violating Riemann:
        - g = 9.81
        - length = 25
        - num_cells = 1000
        - time_final = 1.0
        - num_steps = 10000
        - TEST_CASE = "ENTROPY_RIEMANN"
        - TOPOGRAPHY = "FLAT"

    Positivity violating Riemann:
        - g = 9.81
        - length = 10
        - num_cells = 3000
        - time_final = 0.125
        - num_steps = 10000
        - TEST_CASE = "POSITIVITY_RIEMANN"
        - TOPOGRAPHY = "FLAT"

    Thacker:
        - TOPOGRAPHY = "THACKER"

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


#Gravity coefficient
g           = 9.81

#Mesh [-length,length]
length      = 10
num_cells   = 3000
dx          = 2 * length / num_cells
x           = jnp.linspace(-length + 0.5 * dx, length - 0.5 * dx, num_cells, dtype = DTYPE)

#Temporal
time_final  = 0.125
num_steps   = 20000
dt          = time_final / num_steps

#Boundary
#padding width necessary for implementing boundary conditions
pad_width_flux          = 1
pad_width_diss          = 2
pad_width_source        = 1

num_ghost_cells_flux    = 2 * pad_width_flux + num_cells
num_ghost_cells_diss    = 2 * pad_width_diss + num_cells

#Initial conditions
HEIGHT_IS_FREE_SURFACE = True
initial_condition_params = (20.0, 15.0, 0.0, 0.0) # (2.0, 1.0, 0.0, 0.0) # 

#Bottom topography
topography_params = () # (0.5, 1) # (8.0, 1500 / 4)

#Numerics
TEST_CASE   = "POSITIVITY_RIEMANN"
FLUX        = "FJORDHOLM"
LIMITER     = "MINMOD"
DISSIPATION = "TECNO_ROE"
INTEGRATOR  = "RK4"
BOUNDARY    = "TRANSMISSIVE"
SOURCE      = "FJORDHOLM"
TOPOGRAPHY  = "FLAT"

#data
sample_rate = 10
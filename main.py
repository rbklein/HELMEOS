"""
    Main file containing a JAX-based solution algorithm of the compressible Euler equations with real gas thermodynamics
"""

from setup import *

import FOM
import plot

u = jnp.copy(u_0)

plot.plot_conserved(x, u)
plot.plot_thermodynamic(x, u)
plot.plot_entropy(x, u)

plot.plot_pv(u, True)
plot.show()


time_index = 0
while time_index < num_steps:
    u = step(u, FOM.dudt)

    time_index += 1
    print(time_index * dt)


plot.plot_conserved(x, u)
plot.plot_thermodynamic(x, u)
plot.plot_pv(u, True)
plot.show()


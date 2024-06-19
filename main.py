"""
    Main file containing a JAX-based solution algorithm of the shallow water equations on a periodic domain
"""

from setup import *

import FOM
import plot

u = jnp.copy(u_0)

plot.plot_conserved(x, u)
plot.plot_thermodynamic(x, u)
plot.plot_entropy(x, u)
plot.show()


time_index = 0
while time_index < num_steps:
    u = step(u, FOM.dudt)

    time_index += 1
    print(time_index * dt)


plot.plot_conserved(x, u)
plot.plot_thermodynamic(x, u)
plot.show()


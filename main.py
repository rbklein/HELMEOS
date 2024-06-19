"""
    Main file containing a JAX-based solution algorithm of the shallow water equations on a periodic domain
"""

from setup import *

import FOM
import plot
import data

u = jnp.copy(u_0)

plot.plot_all(x, u)
plot.show()

snapshots = data.prepare_data()
snapshots = data.gather_data(snapshots, u, 0)

time_index = 0
while time_index < num_steps:
    u = step(u, FOM.dudt)

    time_index += 1
    print(time_index * dt)

    snapshots = data.gather_data(snapshots, u, time_index)

plot.plot_all(x, u)
plot.show()

jnp.save("snapshot_data", snapshots)
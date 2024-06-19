"""
    Sets up thermodynamics functions that can be used within all discretization related modules
"""

from config_discretization import *

import thermodynamics

#Thermodynamics functions have to be defined before any module that imports the entropy module to prevent circular imports
eos             = thermodynamics.return_eos(EOS)
pressure        = thermodynamics.generate_pressure(eos)
internal_energy = thermodynamics.generate_internal_energy(eos)
entropy         = thermodynamics.generate_entropy(eos)
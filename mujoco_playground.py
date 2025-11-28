# mujoco_playground.py 

""" More for playing w/ MJC-related DMPC sim tasks. """

# -------------------------------------------------------------------------------------------------

import numpy as np 
import matplotlib.pyplot as plt

from DMPC_Sim import *
from worlds import *
from agents import *

from scenarios import *

# -------------------------------------------------------------------------------------------------

sim = run_ring_formation_scenario()

sim.plot_trajectories()
sim.plot_states()
sim.plot_controls()

a = 0
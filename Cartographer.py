import matplotlib
import matplotlib.pyplot    as     plt
from   matplotlib.animation import FuncAnimation
matplotlib.use('TkAgg')     # Use stable backend for Windows

# -------------------------------------------------------------------------------------------------

class Cartographer:
    """
    
        Handles all animation and visualization. 
    
    """

    def __init__(self):
        pass


    # --- --- --- --- --- CORE --- --- --- --- --- 

    # Animation 
    def animate(self, agents=None): pass 

    # Final Trajectory Plot
    def plot_trajectories(self, agents=None): pass

    # States
    def plot_states(self, agents=None): pass

    # Controls
    def plot_controls(self, agents=None): pass


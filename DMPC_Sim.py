import numpy   as np 
import casadi  as ca
import control as ct

from Oracle import Oracle

# -------------------------------------------------------------------------------------------------

class DMPC_Sim():
    """ 
        
        Simulation executive. 
    
        Instantiates the superstructure and all agents 
        in the environment, and handles synchronous stepping. 
        
    """

    def __init__(self):

        self.Oracle = Oracle()


    # --- --- --- --- --- AGENT MANAGEMENT --- --- --- --- ---

    def add_agent(self, agent):
        """ Add an agent to the simulation. """

        self.Oracle.add_agent(agent.id, agent.x_current)
    

    # --- --- --- --- --- SIMULATION STEPS --- --- --- --- ---

    def step(self):
        """ Step all agents in the simulation. """
        for agent in self.agents:
            agent.step()


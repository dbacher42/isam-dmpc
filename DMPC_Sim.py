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
        """
           Step all agents in the simulation, ensuring synchronous updates.
        """

        data   = self.Oracle.data
        t_hist = self.Oracle.t_hist

        # Part 1: Collect all control inputs 
        for agent in self.agents:
            u = agent.step_control()
            data[agent.name]['control'].append(u)
            
        # Part 2: Propagate all states
        for agent in self.agents:
            x = agent.step_state()
            data[agent.name]['state'].append(x)

        # Time history stored in Oracle
        t_hist.append(t_hist[-1] + self.dt)


# -------------------------------------------------------------------------------------------------

if __name__ == "__main__":

    from model      import FloatbotModel
    from mpc        import FloatbotMPC
    from sim        import FloatbotSim


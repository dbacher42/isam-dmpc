import numpy   as np 
import casadi  as ca
import control as ct

from Oracle import Oracle
from Agent  import Agent

# -------------------------------------------------------------------------------------------------

class DMPC_Sim():
    """ 
        
        Simulation executive. 
    
        Instantiates the superstructure and all agents 
        in the environment, and handles synchronous stepping. 
        
    """

    def __init__(self, dt):

        self.dt     = dt 
        self.agents = []
        self.Oracle = Oracle()


    # --- --- --- --- --- AGENT MANAGEMENT --- --- --- --- ---

    def build_agent(self, name, model, x0, controller_type='MPC', **controller_params):
        """
            Build and return an agent with the specified system model, 
            initial state, and controller + parameters. 

            Crucially, enforces a common dt across all agents. 
        """

        agent = Agent(model, self.dt, x0, name=name)
        agent._build_controller(controller_type, controller_params)
        self.add_agent(agent)
        return agent
    
    # ---

    def add_agent(self, agent):
        """
            Add an agent to the simulation and track
            data in the Oracle.
        """

        self.agents.append(agent)
        self.Oracle.add_agent(agent)


    # --- --- --- --- --- SIMULATION STEPS --- --- --- --- ---

    def run(self, Tf):
        """
            Run the simulation for the specified final time Tf.
        """
        N_steps = int(Tf / self.dt)
        for _ in range(N_steps):
            self.step()

    # --- 

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

        # Time history stored in Oracle too
        t_hist.append(t_hist[-1] + self.dt)


# -------------------------------------------------------------------------------------------------

if __name__ == "__main__":

    from model      import FloatbotModel
    from mpc        import FloatbotMPC
    from sim        import FloatbotSim


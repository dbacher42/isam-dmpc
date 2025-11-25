# DMPC_Sim.py - Simulation Executive

from agents         import *
from Oracle         import Oracle
from viz            import Cartographer
from superstructure import *

import mujoco

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
        self.ready  = False

        self.Oracle       = Oracle()
        self.Cartographer = Cartographer()

        self.env    = mujoco.MjSpec()

    # --- 

    def finalize(self):
        """
            Finalize simulation setup after structure and all agents have been added.
            Perform consistency checks and apply standards across agents. 
        """

        # Compile  
        self.model = self.env.compile()
        self.data  = mujoco.MjData(self.model)

        # Formatting 
        self.Cartographer._assign_colors(self.agents)

        # Done
        self.ready = True


    # --- --- --- --- --- SUPERSTRUCTURE --- --- --- --- ---

    def build_superstructure(self, blocks, connections): a = 0




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
            Add an agent to the simulation and track data in the Oracle.
        """

        self.agents.append(agent)
        self.Oracle.add_agent(agent)
        self.ready = False 


    # --- 

    def _check_ready(self):
        if not self.ready: 
            raise RuntimeError("Simulation not finalized. Call finalize() before running.")
        

    # --- --- --- --- --- SIMULATION STEPS --- --- --- --- ---

    def run(self, Tf):
        """
            Run the simulation for the specified final time Tf.
        """
        self._check_ready()
        
        N_steps = int(Tf / self.dt)
        for _ in range(N_steps):
            self._step()

    # --- 

    def _step(self):
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


    # --- --- --- --- --- PLOTTING WRAPPERS --- --- --- --- ---

    def animate(self, agents=None):
        self._check_ready()
        if agents is None: agents = self.agents
        self.Cartographer.animate(agents)
    
    def plot_trajectories(self, agents=None):
        self._check_ready()
        if agents is None: agents = self.agents
        self.Cartographer.plot_trajectories(agents)

    def plot_states(self, agents=None, layered=True):
        self._check_ready()
        if agents is None: agents = self.agents
        self.Cartographer.plot_states(agents, layered=layered)

    def plot_controls(self, agents=None):
        self._check_ready()
        if agents is None: agents = self.agents
        self.Cartographer.plot_controls(agents)



# -------------------------------------------------------------------------------------------------

if __name__ == "__main__": 
    a=0
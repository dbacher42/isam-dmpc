# DMPC_Sim.py - Simulation Executive

from agents         import *
from superstructure import *
from worlds         import *
from Oracle         import Oracle
from viz            import Cartographer

import mujoco
from   mujoco import viewer

# -------------------------------------------------------------------------------------------------

class DMPC_Sim():
    """ 
        
        Simulation executive. 
    
        Instantiates the superstructure and all agents 
        in the environment, and handles synchronous stepping. 
        
    """

    # --- --- --- --- --- SIMULATION CORE --- --- --- --- ---

    def __init__(self, dt):

        self.dt     = dt 
        self.agents = []
        self.ready  = False

        self.Oracle       = Oracle()
        self.Cartographer = Cartographer()

        self._start_env()

    # ---

    def _start_env(self):
        """ Starts up base MJC spec - does not compile here. """

        self.env = mujoco.MjSpec()
        
    # --- 

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

        truth  = self.Oracle.data
        t_hist = self.Oracle.t_hist

        # Part 1: Collect all control inputs 
        for agent in self.agents:
            u = agent.step_control()
            agent._apply_mjc_control(self.data)          # self.data is from MJC sim  
            truth[agent.name]['control'].append(u)
        
        # Part 2: Syncrhonous MJC update 
        mujoco.mj_step(self.model, self.data)

        # Part 3: Extract and update states locally in agent + Oracle  
        for agent in self.agents:
            x = agent._extract_mjc_state(self.data)
            truth[agent.name]['state'].append(x)

        # Time history stored in Oracle too
        t_hist.append(t_hist[-1] + self.dt)

    # --- 

    def _check_ready(self):
        if not self.ready: 
            raise RuntimeError("Simulation not finalized. Call finalize() before running.")
        
    
    # --- --- --- --- --- MUJOCO COMPILE AND VIEW --- --- --- --- ---

    def finalize(self):
        """
            Finalize simulation setup after structure and all agents have been added.
            Perform consistency checks and apply standards across agents. 
        """

        # Compile  
        self.model = self.env.compile()
        self.data  = mujoco.MjData(self.model)

        # Initialize all agent states from their x0 values
        for agent in self.agents:
            agent._set_mjc_initial_state(self.data)

        # Formatting 
        self.Cartographer._assign_colors(self.agents)

        # Done
        self.ready = True

    # ---

    def _view_mjc(self):
        """ MuJoCo viewer for current sim. """

        if not self.ready:
            raise RuntimeError("Simulation not finalized. Call finalize() before viewing.")
        
        viewer.launch(self.model, self.data)


    # --- --- --- --- --- SUPERSTRUCTURE --- --- --- --- ---

    def build_superstructure(self, blocks, connections): 
        
        # Build
        structure = Superstructure_2D()
        STRUCTURE = structure.generate_model(blocks, connections)
        
        # Add to spec 
        self.env.attach(STRUCTURE, frame=self.env.worldbody.add_frame())
        self.ready = False 


    # --- --- --- --- --- AGENT MANAGEMENT --- --- --- --- ---

    def build_agent(self, name, model, x0, controller_type='MPC', **controller_params):
        """
            Build and return an agent with the specified system model, 
            initial state, and controller + parameters. 

            Crucially, enforces a common dt across all agents. 
        """

        agent = Agent(model, self.dt, x0, name=name)
        agent.build_controller(controller_type, controller_params)
        self.add_agent(agent)
        return agent
    
    # ---

    def add_agent(self, agent):
        """
            Add an agent to the simulation and track data in the Oracle.
        """

        # Sim
        self.agents.append(agent)
        self.Oracle.add_agent(agent)
        
        # Add to MJC 
        self.env.attach(agent.spec, frame=self.env.worldbody.add_frame())
        self.ready = False 




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

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
        self.last_control_time = 0

        self.Oracle       = Oracle()
        self.Cartographer = Cartographer()

        self._start_env()
        
    # --- 

    def run(self, time, render=True, real_time=True):
        """
            Run the simulation for the specified time.
            
            Parameters
            ----------
            time : float
                Simulation time
            render : bool, optional 
                Whether to show real-time rendering (default: True)
            real_time : bool, optional
                Whether to run at real-time speed (default: True)
        """
        import time as time_module
        
        self._check_ready()
        
        if render:
            self._start_viewer()
        
        start_time = time_module.time()
        
        step_times = []
        
        i = 0
        while self.data.time <= time:
            step_start = time_module.time()
            
            self._step()
            
            elapsed = time_module.time() - step_start
            step_times.append(elapsed)
            
            # Enforce real-time if desired
            if real_time:
                time_module.sleep(.002) # default MuJoCo time step

            i += 1
        
        if render:
            # Keep viewer open a bit longer
            time_module.sleep(2)
            self._stop_viewer()
            
        total_time = time_module.time() - start_time
        print(f"Simulation completed: {time}s simulated in {total_time:.2f}s wall time")

    # --- 

    def _step(self):
        """
           Step all agents in the simulation, ensuring synchronous updates.
        """

        truth  = self.Oracle.data
        t_hist = self.Oracle.t_hist

        # Part 1: Run agent controllers every dt seconds
        if self.data.time >= self.last_control_time + self.dt:
            for agent in self.agents:
                u = agent.step_control()
                agent._apply_mjc_control(self.data)          # self.data is from MJC sim  
                truth[agent.name]['control'].append(u)
            self.last_control_time = self.data.time
        
        # Part 2: Synchronous MJC update 
        mujoco.mj_step(self.model, self.data)

        # Part 3: Extract and update states locally in agent + Oracle  
        for agent in self.agents:
            x = agent._extract_mjc_state(self.data)
            truth[agent.name]['state'].append(x)

        # Time history stored in Oracle too
        t_hist.append(self.data.time)

        # Render if viewer is active
        if self.viewer is not None:
            self.viewer.sync()

    # --- 

    def _check_ready(self):
        if not self.ready: 
            raise RuntimeError("Simulation not finalized. Call finalize() before running.")
        
    
    # --- --- --- --- --- MUJOCO COMPILE AND VIEW --- --- --- --- ---

    def _start_env(self):
        """ Starts up base MJC spec - does not compile here. """

        self.env = mujoco.MjSpec()

    # --- 

    def finalize(self, verbose=False):
        """
            Finalize simulation setup after structure and all agents have been added.
            Perform consistency checks and apply standards across agents. 
        """

        # Compile  
        self.model = self.env.compile()
        self.data  = mujoco.MjData(self.model)

        # Initialize all agent states from their x0 values
        print(f"Setting initial states for {len(self.agents)} agents...")
        for agent in self.agents:
                agent._set_mjc_initial_state(self.data)

        # Verbose
        if verbose:
            print("=== BODY POSITIONS ===")
            for agent in self.agents:
                body_name = f"{agent.name}_floatbot"
                body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
                if body_id >= 0:
                    pos = self.data.xpos[body_id]
                    print(f"{agent.name}: {pos}")

        # Formatting 
        self.Cartographer._assign_colors(self.agents)

        # Optional viewer 
        self.viewer = None

        # Done
        self.ready = True

    # ---

    def _start_viewer(self):
        """ Start real-time viewer."""
        
        if not self.ready:
            raise RuntimeError("Simulation not finalized. Call finalize() before starting viewer.")
        
        if self.viewer is None:
            self.viewer = viewer.launch_passive(self.model, self.data)
    
    # --- 

    def _stop_viewer(self):
        """ Stop real-time viewer. """
        
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


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

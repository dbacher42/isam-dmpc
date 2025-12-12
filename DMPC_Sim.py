# DMPC_Sim.py - Simulation Executive

import numpy as np

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
        self.viewer       = None

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
        if self.data.time >= self.last_control_time + self.dt:
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

    def compile(self):
        """ 
            Compile the MuJoCo model for the entire simulation. 

            Must be called before running simulation, after all 
            agents and structures have been added.

            Agent initial states and docking configurations
            are set in finalize(). 
        """

        self.model = self.env.compile()
        self.data  = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

    # --- 

    def finalize(self):
        """
            Finalize simulation setup after structure and all agents have been added.
            Perform consistency checks and apply standards across agents. 
        """

        # Compile  
        self.compile()

        # Initialize agent states from their x0 values
        # Skip docked agents - their position is determined by the weld constraint
        print(f"Setting initial states for {len(self.agents)} agents...")
        for agent in self.agents:
            if hasattr(agent, 'docked_to') and agent.docked_to is not None:
                print(f"  Skipping {agent.name} (docked to {agent.docked_to})")
            else:
                agent._set_mjc_initial_state(self.data)

        # Check agents have Z offset if structure present
        if hasattr(self, 'structure'):
            if any(agent.z_offset == 0 for agent in self.agents):
                print('WARNING: Some agents are in-plane with the structure and may collide.\n' \
                      'Set a target_z for proper offset and docking function first.') 

        # Formatting 
        self.Cartographer._assign_colors(self.agents)

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
        self.structure = Superstructure_2D()
        self.structure.generate_model(blocks, connections)
        
        # Add to spec 
        self.env.attach(self.structure.spec, frame=self.env.worldbody.add_frame())
        self.ready = False 

    # ---

    def get_block_positions(self, block_id=None):
        """
        Get current position(s) of superstructure blocks from simulation data.
        
        Parameters
        ----------
        block_id : str, optional
            Specific block ID to query. If None, returns all blocks.
            
        Returns
        -------
        dict or np.array
            If block_id is None: {block_id: np.array([x, y, z])} for all blocks
            If block_id specified: np.array([x, y, z]) for that block
        """
        if not hasattr(self, 'data'):
            raise RuntimeError("Must call compile() before querying block positions")
        if not hasattr(self, 'structure'):
            raise RuntimeError("No structure present.")
            
        if block_id is not None:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, block_id)
            if body_id < 0:
                raise ValueError(f"Block '{block_id}' not found in model")
            return self.data.xpos[body_id].copy()
        
        # All blocks by default
        positions = {}
        for bid in self.structure.blocks:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, bid)
            if body_id >= 0:
                positions[bid] = self.data.xpos[body_id].copy()
        return positions

    # ---

    def get_attachment_points(self, block_id=None):
        """
             Get attachment points for docking agents below structure blocks.

             NOTE: will be expanded in the future once agents can dock at different faces 
                   / exist in-plane with the structure rather than below it. For now, agents
                   just dock below each block. 
                   
             Parameters
             ----------
             block_id : str, optional
                 Specific block ID to query. If None, returns all blocks.
                 
             Returns
             -------
             dict or np.array
                 Block positions with Z set to bottom face (for docking target)
        """
        positions = self.get_block_positions(block_id)
        
        # Target Z is block bottom face
        if block_id is not None:
            # Single block - positions is already np.array
            block_half_height = self.structure.blocks[block_id].size / 2
            target_z = positions[2] - block_half_height
            return np.array([positions[0], positions[1], target_z])
        
        # All blocks - positions is dict
        return {bid: np.array([pos[0], pos[1], pos[2] - self.structure.blocks[bid].size / 2]) 
                for bid, pos in positions.items()}


    # --- --- --- --- --- AGENT MANAGEMENT --- --- --- --- ---

    def build_agent(self, name, model, x0, controller_type='MPC', target_z=None, dock_block=None, **controller_params):
        """
            Build and return an agent with the specified system model, 
            initial state, and controller + parameters. 

            Crucially, enforces a common dt across all agents. 

            target_z is where the top of the agent should align to for docking. The agent
            internally computes its own z_offset based on its voxel geometry to make sure
            that its top face matches target_z.
            
            dock_block is the block ID to dock to (optional). If provided, the agent is
            attached at the block's XY position and a weld constraint is created.
        """

        agent = Agent(model, self.dt, x0, name=name)
        agent.build_controller(controller_type, controller_params)
        self.add_agent(agent, target_z=target_z, dock_block=dock_block)
        return agent
    
    # ---

    def add_agent(self, agent, target_z=None, dock_block=None):
        """
            Add an agent to the simulation and track data in the Oracle.
            
            If dock_block is specified, attaches agent at block's XY position
            and creates a weld constraint.
        """

        # Sim
        self.agents.append(agent)
        self.Oracle.add_agent(agent)
        
        # Compute z_offset if not already set (default: dock to z=0 plane)
        if target_z:
            agent.compute_docking_z_offset(target_z)
        
        # Determine attachment position
        if dock_block is not None:
            # Get block position for XY, use agent z_offset for Z
            block_pos = self.get_block_positions(dock_block)
            attach_pos = [block_pos[0], block_pos[1], agent.z_offset]
        else:
            attach_pos = [0, 0, agent.z_offset]
        
        # Add to MJC 
        self.env.attach(agent.spec, frame=self.env.worldbody.add_frame(pos=attach_pos))
        self.ready = False
        
        # Create weld constraint for docking after adding to spec (in prev step) 
        if dock_block is not None:
            self._create_dock_constraint(agent, dock_block) 

    # ---
    
    def _create_dock_constraint(self, agent, block_id):
        """
            Internal: Create a weld constraint between an agent's voxel and a structure block.
        """
        # Constraint name
        constraint_name = f"dock_{agent.name}"
        
        # Target body
        voxel_body = f"{agent.name}_voxel"
        
        # Add weld equality constraint to spec
        eq = self.env.add_equality(
            name=constraint_name,
            objtype=mujoco.mjtObj.mjOBJ_BODY
        )
        eq.type = mujoco.mjtEq.mjEQ_WELD
        eq.name1 = block_id                  
        eq.name2 = voxel_body               
        eq.active = True                    
        
        # Log
        agent.docked_to = block_id
        agent.dock_constraint_name = constraint_name
        print(f'Docked agent {agent.name} to block {block_id} with constraint {constraint_name}')

    # ---

    def dock_agent(self, agent, block_id):
        """
            Create a weld constraint between an agent's voxel and a structure block.
            
            NOTE: Prefer using dock_block parameter in build_agent() instead.
            This method is for cases where agent is already added but needs docking.
            
            Must be called AFTER add_agent() and BEFORE finalize().
            The weld locks the relative pose as defined at model compile time,
            so agent must already be positioned correctly via z_offset.
            
            WARNING: Does not update agent attachment position - agent must already
            be at correct XY position. Use build_agent(dock_block=...) for automatic
            positioning.
            
            Parameters
            ----------
            agent : Agent
                The agent to dock
            block_id : str
                ID of the structure block to dock to (e.g., "block_0_2")
        """
        if self.ready:
            raise RuntimeError("Cannot dock after finalize(). Call dock_agent() before finalize().")
        
        self._create_dock_constraint(agent, block_id)


    # --- --- --- --- --- PARAMETER ESTIMATION --- --- --- --- ---

    def get_composite_mass_properties(self):
        """
        Compute composite mass properties for entire system (structure + all agents).
        
        Returns mass, CG (world frame), and inertia for the composite system
        by summing contributions from all bodies in the simulation.
        
        Returns
        -------
        mass : float
            Total composite mass (kg)
        cg : np.array (2,)
            Composite CG in world frame [x, y] (m)
        inertia : float
            Composite z-axis inertia (kg*m^2)
        """
        if not self.ready:
            raise RuntimeError("Must call finalize() and run simulation before extracting mass properties")
        
        # Collect masses and positions for all bodies (structure + all agents)
        masses = []
        positions = []
        inertias_local = []
        
        for i in range(self.model.nbody):
            body_name = self.model.body(i).name
            if body_name == 'world':
                continue
            masses.append(self.model.body_mass[i])
            positions.append(self.data.xpos[i][:2])
            inertias_local.append(self.model.body_inertia[i][2])  # Izz
        
        # Composite mass
        masses = np.array(masses)
        positions = np.array(positions)
        inertias_local = np.array(inertias_local)
        composite_mass = masses.sum()
        
        # Composite CG
        composite_cg = (masses[:, np.newaxis] * positions).sum(axis=0) / composite_mass
        
        # Composite inertia (parallel axis theorem: I = sum(I_local_i + m_i * d_i^2))
        d_sq = ((positions - composite_cg)**2).sum(axis=1)  # distance^2 from composite CG
        composite_inertia = (inertias_local + masses * d_sq).sum()
        
        return composite_mass, composite_cg, composite_inertia


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

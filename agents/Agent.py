# Agent.py 

import os 
import numpy as np
import mujoco 

from .controllers import *

# -------------------------------------------------------------------------------------------------

class Agent():
    """
    
        Builds and simulates a single floatbot agent. 

        Can read/write to the Oracle during sim. 
    
    """

    def __init__(self, model, dt, x0=np.array([0, 0, 1, 0, 0, 0, 0]), name=None):
        """
            Parameters
            ----------
                model : floatbot kinematics and dynamics model
                x0 : initial states
                    optional, the default is np.array([0, 0, 1, 0, 0, 0, 0]).
        """

        # Agent parameters
        self.name       = name
        self.model      = model
        self.dt         = dt

        # Initialize data lists
        self.t_history = [0]
        self.x_history = [x0]
        self.u_history = []
        
        # Current 
        self.x_current = x0
        self.u_current = np.zeros(8)

        # MJC
        self._build_mjc_model()
    
    # --- 

    def build_controller(self, controller_type, controller_params):
        """
            Build controller internally with provided dt
        """

        if 'x_cmd' not in controller_params:
            raise ValueError("Controller must be provided a reference state 'x_cmd'")
        x_cmd = controller_params['x_cmd']

        if controller_type == 'MPC':
            H = controller_params.get('H', 20)
            Q = controller_params.get('Q', np.eye(6))
            R = controller_params.get('R', np.eye(4))
            self.controller = FloatbotMPC(self.model, x_cmd, self.dt, H, Q, R)

        elif controller_type == 'LQR':
            Q = controller_params.get('Q', np.eye(6))
            R = controller_params.get('R', np.eye(4))
            self.controller = FloatbotLQR(self.model, x_cmd, self.dt, Q, R)


    # --- --- --- --- --- SIMULATION STEP --- --- --- --- ---

    def step_control(self):
        """
            Compute control input for the current state
        """

        if not hasattr(self, 'controller'):
            raise ValueError("Controller has not been built yet. Call _build_controller() first.")

        # Extract parameters
        control_law = self.controller.solve
        
        # Compute control input
        self.u_current = control_law(self.x_current)
        self.u_history.append(self.u_current)
        return self.u_current

    # ---

    def step_state(self):
        """
            Propagate state forward one time step
        """
        # Extract parameters
        dt   = self.dt
        xdot = self.model.xdot
        
        # Propogate state forward
        x_next = self.x_current + dt*xdot(self.x_current, self.u_current)
        self.x_current = np.array(x_next.full()).flatten()
        
        # Normalize quaternion
        qw = self.x_current[2]
        qz = self.x_current[3]
        q_mag = np.sqrt(qw**2+qz**2)
        self.x_current[2] = qw/q_mag
        self.x_current[3] = qz/q_mag
        
        self.x_history.append(self.x_current)
        self.t_history.append(self.t_history[-1] + dt)
        return self.x_current


    # --- --- --- --- --- MUJOCO --- --- --- --- ---

    def _build_mjc_model(self):
        """ 
            Build MuJoCo model for this agent. 
        """
        
        path = os.path.join(os.path.dirname(__file__), 'mujoco_agent.xml')
        spec = mujoco.MjSpec.from_file(path)
        base = spec.worldbody.first_body()
        base.name = self.name

        # Store spec now, compile in main sim later 
        self.spec = spec 
        self._rename_spec()

    # --- 

    def _check_ready(self):
        if not hasattr(self, 'model'):
            raise RuntimeError("MuJoCo model not built yet. Call _build_mjc_model() first.")
        if not hasattr(self, 'name'):
            raise RuntimeError("Must set agent.name first.")
        
    # ---

    def _apply_mjc_control(self, data):
        """
            Apply current control input to MuJoCo actuators. 
        """

        self._check_ready()

        # Apply control inputs
        for i in range(4):                                            # NOTE: hardcoded 4 thrusters 
            actuator_name = f"{self.name}_thruster_{i+1}"
            data.ctrl[actuator_name] = self.u_current[i]

    # --- 

    def _set_mjc_initial_state(self, data):
        """
            Set initial joint positions and velocities from x0.
        """
        
        # Get initial state
        x0 = self.x_history[0]
        rx, ry = x0[0], x0[1]
        qw, qz = x0[2], x0[3]
        vx, vy, wz = x0[4], x0[5], x0[6]
        
        # Quat to angle 
        tht = 2 * np.arctan2(qz, qw)
        
        # Set joint positions
        X_joint   = data.joint(f"{self.name}_x")
        Y_joint   = data.joint(f"{self.name}_y") 
        THT_joint = data.joint(f"{self.name}_theta")
        
        X_joint.qpos[0]   = rx
        Y_joint.qpos[0]   = ry  
        THT_joint.qpos[0] = tht
        
        # Set joint velocities
        X_joint.qvel[0]   = vx
        Y_joint.qvel[0]   = vy
        THT_joint.qvel[0] = wz

    # --- 

    def _extract_mjc_state(self, data):
        """
            Extract current state from MuJoCo data. 
        """
        
        # Read from joints directly 
        X   = data.joint(f"{self.name}_x")
        Y   = data.joint(f"{self.name}_y") 
        THT = data.joint(f"{self.name}_theta")
        
        rx  =   X.qpos[0]
        ry  =   Y.qpos[0]
        tht = THT.qpos[0]
        vx  =   X.qvel[0]
        vy  =   Y.qvel[0]
        wz  = THT.qvel[0]
        
        # Convert angle to quaternion
        qw = np.cos(tht/2)
        qz = np.sin(tht/2)
        
        self.x_current = np.array([rx, ry, qw, qz, vx, vy, wz])
        return self.x_current

    # ---

    def _rename_spec(self):
        """
            Add unique agent name to all entities in the spec to avoid conflicts.
        """

        self._check_ready()
        spec   = self.spec
        prefix = self.name
        
        # ---

        def _rename(body):
            """Recursively prefix body and all its children."""
            if body.name:
                body.name = f"{prefix}_{body.name}"
            
            # Geoms
            for geom in body.geoms:     
                if geom.name:
                    geom.name = f"{prefix}_{geom.name}"

            # Sites
            for site in body.sites:
                if site.name:
                    site.name = f"{prefix}_{site.name}"
            
            # Joints
            for joint in body.joints:
                if joint.name:
                    joint.name = f"{prefix}_{joint.name}"
                    
            # Recurse
            for child_body in body.bodies:
                _rename(child_body)
        
        # --- 
        
        # Rename all bodies 
        for body in spec.worldbody.bodies:
            _rename(body)
        
        # Actuators
        for actuator in spec.actuators:
            if actuator.name:
                actuator.name = f"{prefix}_{actuator.name}"
            # Update site reference
            if hasattr(actuator, 'site') and actuator.site:
                actuator.site = f"{prefix}_{actuator.site}"


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
        self.z_offset   = 0          # For docking underneath blocks 

        # Initialize data lists
        self.t_history = [0]
        self.x_history = [x0]
        self.u_history = []
        
        # Current 
        self.x_current = x0
        self.u_current = np.zeros(8)

        # MJC - also extracts geometry from XML
        self._build_mjc_model()

    # ---

    def compute_docking_z_offset(self, target_z):
        """
            Compute Z offset for docking below a target plane/position.
            
            Places agent such that voxel top face aligns with target_z.
            
            Parameters
            ----------
            target_z : float
                Z position of the docking surface (e.g., block bottom face)
                Default 0.0 assumes block center at origin with half-height 0.5,
                so block bottom is at -0.5. For that case, pass target_z=-0.5.
                
            Returns
            -------
            float
                Z offset for agent frame placement
        """

        # Agent voxel top needs to align with target_z
        voxel_top_local = self.voxel_local_z + self.voxel_half_height
        self.z_offset = target_z - voxel_top_local
        return self.z_offset
    
    # --- 

    def build_controller(self, controller_type, controller_params):
        """
            Build controller internally with provided dt
        """

        if 'x_cmd' not in controller_params:
            raise ValueError("Controller must be provided a reference state 'x_cmd'")
        x_cmd = controller_params['x_cmd']

        if controller_type not in ['LQR','Basic_MPC','Excitation_MPC']:
            raise ValueError(f"Unknown controller type '{controller_type}'")

        if controller_type == 'Basic_MPC':
            H = controller_params.get('H', 20)
            Q = controller_params.get('Q', np.eye(6))
            R = controller_params.get('R', np.eye(4))
            max_thrust = controller_params.get('max_thrust', 1.5)
            self.controller = Basic_MPC(self.model, x_cmd, self.dt, H, Q, R, max_thrust)

        elif controller_type == 'Excitation_MPC':
            H = controller_params.get('H', 20)
            Q = controller_params.get('Q', np.eye(6))
            R = controller_params.get('R', np.eye(4))
            l = controller_params.get('lambda_fim', np.ones((4,1)))
            s = controller_params.get('covariance', 0.01*np.eye(7))
            max_thrust = controller_params.get('max_thrust', 1.5)
            self.controller = Excitation_MPC(x_cmd, self.dt, H, Q, R, l, s, max_thrust)

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
        self.u_current = control_law(self.x_current, self.model)
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

    # ---

    def estimate(self, oracle_data=None, update_model=True):
        """
            Estimate mass properties from trajectory data using least-squares.
            
            Uses trajectory data to build regression:
                b = A @ theta
            
            Where theta = [1/m, rho_x, rho_y, Jzz/m] (transformed parameters)
            
            Parameters
            ----------
            oracle_data : dict, optional
                If provided, uses oracle_data['state'] and oracle_data['control']
                Otherwise uses self.x_history and self.u_history
            update_model : bool
                If True, updates self.model with estimated parameters
                
            Returns
            -------
            tht_hat : 4x1 array
                Estimated parameters [mass, rho_x, rho_y, Jzz]
        """
        
        # Use Oracle data if provided, otherwise use agent's own history
        if oracle_data is not None:
            x_data = oracle_data['state']
            u_data = oracle_data['control']
        else:
            x_data = self.x_history
            u_data = self.u_history
        
        def rot(qw, qz):
            """2D rotation matrix from quaternion"""
            return np.array([
                [qw**2 - qz**2, -2*qw*qz],
                [2*qw*qz, qw**2 - qz**2]
            ])
        
        def mixer(u, R):
            """Compute body forces from control inputs"""
            f_B = np.array([u[0] + u[2], u[1] + u[3]])
            f_I = R @ f_B
            tau = self.model.moment_arm * (-u[0] - u[1] + u[2] + u[3])
            return np.array([f_I[0], f_I[1], tau])
        
        # Number of data points (need at least 2 for acceleration)
        N = len(u_data)
        if N < 2:
            raise ValueError("Need at least 2 timesteps of data to estimate")
        
        # Debug: check data shapes
        print(f"DEBUG: N = {N}")
        print(f"DEBUG: x_data[0] = {x_data[0]}")
        print(f"DEBUG: x_data[1] = {x_data[1]}")
        print(f"DEBUG: x_data[4] = {x_data[4]}")
        print(f"DEBUG: x_data[8] = {x_data[8]}")
        print(f"DEBUG: x_data[-1] = {x_data[-1]}")
        print(f"DEBUG: u_data[0] = {u_data[0]}")
        
        # Build regression matrices
        b = np.zeros((3*N, 1))
        A = np.zeros((3*N, 4))
        
        dt = self.dt
        
        for k in range(N):
            # Current state
            x_k = np.array(x_data[k])
            qw, qz = x_k[2], x_k[3]
            wz = x_k[6]
            
            # Compute acceleration via finite difference
            x_next = np.array(x_data[k+1])
            acc = (x_next - x_k) / dt  # [px_dot, py_dot, qw_dot, qz_dot, vx_dot, vy_dot, wz_dot]
            
            # Extract velocity derivatives (accelerations)
            vx_dot = acc[4]
            vy_dot = acc[5]
            wz_dot = acc[6]
            
            # Rotation matrix and forces
            R = rot(qw, qz)
            F = mixer(np.array(u_data[k]), R)  # F = [fx_inertial, fy_inertial, tau]
            
            # Fill regression matrices (following teammate's formulation)
            # Row indices for this timestep
            i0, i1, i2 = 3*k, 3*k+1, 3*k+2
            
            # b vector (measurements - inertial frame accelerations)
            b[i0, 0] = vx_dot
            b[i1, 0] = vy_dot
            
            # A matrix columns: [1/m, rho_x, rho_y, Jzz/m]
            # Column 0: 1/m (multiplies inertial force)
            A[i0, 0] = F[0]
            A[i1, 0] = F[1]
            A[i2, 0] = F[2]
            
            # Column 1: rho_x
            A[i0, 1] = wz**2
            A[i1, 1] = wz_dot
            A[i2, 1] = -vy_dot
            
            # Column 2: rho_y
            A[i0, 2] = wz_dot
            A[i1, 2] = wz**2
            A[i2, 2] = vx_dot
            
            # Column 3: Jzz/m
            A[i2, 3] = -wz_dot
        
        # Solve least squares
        theta = np.linalg.pinv(A) @ b
        
        # Debug: check theta
        print(f"DEBUG: theta = {theta.flatten()}")
        print(f"DEBUG: A column norms = {[np.linalg.norm(A[:,i]) for i in range(4)]}")
        
        # Transform back to physical parameters
        mass = 1.0 / theta[0, 0]
        rho_x = theta[1, 0]
        rho_y = theta[2, 0]
        Jzz = theta[3, 0] / theta[0, 0]  # (Jzz/m) / (1/m) = Jzz
        
        tht_hat = np.array([mass, rho_x, rho_y, Jzz])
        
        # Update model if requested
        if update_model:
            self.model.mass = mass
            self.model.cg = np.array([rho_x, rho_y])
            self.model.inertia = Jzz
        
        return tht_hat


    # --- --- --- --- --- MUJOCO --- --- --- --- ---

    def _build_mjc_model(self):
        """ 
            Build MuJoCo model for this agent and extract geometry. 
        """
        
        path = os.path.join(os.path.dirname(__file__), 'mujoco_agent.xml')
        spec = mujoco.MjSpec.from_file(path)
        
        # Extract voxel geometry from spec
        self._extract_geometry(spec)
        
        # Don't set base.name here - let _rename_spec handle it
        # Store spec now, compile in main sim later 
        self.spec = spec 
        self._rename_spec()
    
    # ---

    def _extract_geometry(self, spec):
        """
            Extract voxel geometry from MjSpec for docking calculations.
        """
        # Find floatbot body and voxel child
        floatbot = spec.worldbody.first_body()
        
        # Find voxel body (child of floatbot)
        voxel_body = floatbot.first_body()
        while voxel_body is not None:
            if 'voxel' in voxel_body.name:
                break
            voxel_body = voxel_body.next_body()
        
        if voxel_body is None:
            raise RuntimeError("Could not find voxel body in agent XML")
        
        # Extract voxel local position (relative to floatbot)
        self.voxel_local_z = voxel_body.pos[2]
        
        # Extract voxel geom half-height (box size[2])
        voxel_geom = voxel_body.first_geom()
        if voxel_geom is None:
            raise RuntimeError("Could not find voxel geom in agent XML")
        
        self.voxel_half_height = voxel_geom.size[2]

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

        # Apply control inputs using name lookup (now that actuators exist)
        for i in range(4):
            actuator_name = f"{self.name}_thruster_{i}"
            actuator_id = mujoco.mj_name2id(data.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
            
            if actuator_id >= 0:
                data.ctrl[actuator_id] = self.u_current[i]
            else:
                print(f"Warning: Actuator {actuator_name} not found")

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
        
        # Update derived quantities + body positions from joints
        mujoco.mj_forward(data.model, data)

    # --- 

    def _extract_mjc_state(self, data):
        """
            Extract current state from MuJoCo data.
            
            For docked agents, reads body position/velocity directly since
            the agent's own joints stay at 0 while the assembly moves via
            structure joints.
        """
        import mujoco
        
        # Check if agent is docked - if so, read body state directly
        if hasattr(self, 'docked_to') and self.docked_to is not None:
            # Get body position and orientation from xpos/xquat
            body_id = mujoco.mj_name2id(data.model, mujoco.mjtObj.mjOBJ_BODY, f"{self.name}_floatbot")
            
            rx, ry, rz = data.xpos[body_id]
            
            # xquat is [w, x, y, z] - extract planar rotation (w, z)
            qw, qx, qy, qz = data.xquat[body_id]
            
            # Get velocity from cvel (6D spatial velocity: [angular, linear])
            # cvel returns [wx, wy, wz, vx_inertial, vy_inertial, vz_inertial]
            # Note: Despite comments in model code, velocities are actually INERTIAL frame
            cvel = data.cvel[body_id]
            wz = cvel[2]  # angular velocity about Z
            vx = cvel[3]  # linear velocity X in inertial frame
            vy = cvel[4]  # linear velocity Y in inertial frame
            
            self.x_current = np.array([rx, ry, qw, qz, vx, vy, wz])
            self.x_history.append(self.x_current)
            return self.x_current
        
        # Not docked - read from joints directly 
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
        self.x_history.append(self.x_current)
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
        
        # Actuators - check what attribute holds site reference
        for actuator in spec.actuators:
            if actuator.name:
                actuator.name = f"{prefix}_{actuator.name}"
            
            # Try different possible attributes for site reference
            for attr in ['site', 'refsite', 'target']:
                if hasattr(actuator, attr):
                    site_ref = getattr(actuator, attr)
                    if site_ref and isinstance(site_ref, str):
                        setattr(actuator, attr, f"{prefix}_{site_ref}")
                        break


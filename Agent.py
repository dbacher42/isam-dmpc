import numpy as np
from lqr import FloatbotLQR
from model import FloatbotModel
from mpc import FloatbotMPC

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
    
    # --- 

    def _build_controller(self, controller_type, controller_params):
        """
            Build controller internally with provided dt
        """

        if 'x_cmd' not in controller_params:
            raise ValueError("Controller must be provided a reference state 'x_cmd'")
        x_cmd = controller_params['x_cmd']

        if controller_type == 'MPC':
            H = controller_params.get('H', 10)
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
        return self.x_current

    # --- 

    def step(self, i):
        """
            Compute control input and propagate state forward one step. 
        """
        
        self.step_control()
        self.step_state()
    
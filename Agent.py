import numpy as np

# -------------------------------------------------------------------------------------------------

class Agent():
    """
    
        Simulates a single floatbot agent with a given
        system model and controller. 

        Can read/write to the Oracle during sim. 
    
    """

    def __init__(self, model, controller, x0=np.array([0, 0, 1, 0, 0, 0, 0])):
        """
            Parameters
            ----------
                model : floatbot kinematics and dynamics model
                controller : floatbot controller
                x0 : initial states
                    optional, the default is np.array([0, 0, 1, 0, 0, 0, 0]).
        """

        # Set simulation parameters
        self.model = model
        self.controller = controller
        self.dt = controller.dt
        
        #Initialize data lists
        self.t_history = [0]
        self.x_history = [x0]
        self.u_history = []
        
        self.x_current = x0
        self.u_current = np.zeros(8)
    

    # --- --- --- --- --- SIMULATION STEP --- --- --- --- ---

    def step_control(self):
        """
            Compute control input for the current state
        """
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
    
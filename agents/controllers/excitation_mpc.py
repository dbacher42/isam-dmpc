import numpy as np
import casadi as cs
import numpy as np

class Excitation_MPC():
    
    def __init__(self, model, x_cmd, dt, H, Q=np.eye(6), R=np.eye(4), lambda_fim=np.ones(4), covariance=0.01*np.eye(7), max_thrust=np.inf):
        '''
        EXCITATION MPC Class 

        Adds information parameters to the controller to generate excitation trajectories. 

        Parameters
        ----------
        model : floatbot kinematics and dynamics model
        x_cmd : commanded states
        dt : time step
        H : time horizon
        Q : weight matrix penalizing state errors
            optional, he default is np.eye(6)
        R : weight matrix penalizing control effort
            optional, the default is np.eye(8)
        covariance : weight vector penalizing information parameters
            optional, the default is np.ones(4)
        bounds : dictionary of bounds on the states in the format:
            'lb_xi' : lower bound on state i
            'ub_xi' : upper bound on state i
            optional, the default is {}
        '''

        # Parameters
        self.model = model
        self.x_cmd = x_cmd
        self.dt = dt
        self.H = H
        self.Q = Q
        self.R = R
        self.lambda_fim = lambda_fim
        self.covariance = covariance
        self.u_max = max_thrust
        self.name  = 'MPC'
        
        # Vector sizes
        self.nx = 7                         # states
        self.nu = 4                         # control inputs
        self.ntht = 4                       # information parameters
        self.nd = self.nx*(H+1) + self.nu*H # decision variables
        self.ng = self.nx*(H+1)             # constraint equations
        
        # Initialize decision variable bounds at +-inf
        self.lbx = -np.inf*np.ones(self.nd)
        self.ubx =  np.inf*np.ones(self.nd)
        
        # Set control bounds
        for i in range(H):
            for j in range(self.nu):
                idx = self.nx*(H+1) + i*self.nu + j
                self.lbx[idx] = -self.u_max
                self.ubx[idx] = self.u_max
                
        # Initialize constraint bounds at 0
        self.lbg = np.zeros(self.ng)
        self.ubg = np.zeros(self.ng)
        
        # Set up the solver
        self.solver = self.setup()
    
    # --- 

    def setup(self):
        '''
        Create the nonlinear OCP solver
        '''
        # Extract parameters
        xdot = self.model.xdot
        X_cmd = cs.DM(self.x_cmd)
        dt = self.dt
        H = self.H
        Q = cs.DM(self.Q)
        R = cs.DM(self.R)
        
        # Initialize Decision variables
        X = cs.SX.sym('X', self.nx, H+1)
        U = cs.SX.sym('U', self.nu, H)

        # Information: partial derivatives of system dynamics w/r/t state 
        x   = cs.SX.sym('x', self.nx, 1)
        u   = cs.SX.sym('u', self.nu, 1)
        tht = cs.SX.sym('tht', self.ntht, 1)
        
        Jx     = cs.jacobian(xdot(x, u, tht), x)
        J_tht  = cs.jacobian(xdot(x, u, tht), tht)
        Jx_k   = cs.Function('Jx'  , [x,u,tht], [Jx]) 
        Jtht_k = cs.Function('Jtht', [x,u,tht], [J_tht])

        # Fisher Information 
        PHI = np.zeros([self.nx, self.ntht]) 
        F   = np.eye(self.ntht)               # init as identity to avoid singularity 

        # Initialize cost function, constraints, and parameters
        cost = 0  # accumulated cost
        g    = [] # list of constraints
        
        # Initial condition constraint: X0 = current state
        X0 = cs.SX.sym('X0', self.nx)
        g.append(X[:,0] - X0)
        
        # Loop over time horizon to build cost function and constraints
        f = lambda x, u: x + dt*xdot(x,u) # Discretized dynamics using Euler integration
        
        for k in range(H):    

            # Error 
            X_err = self.error(X[:,k], X_cmd)
            
            # FIM 
            F += PHI.T @ cs.inv(self.covariance) @ PHI 

            # Error includes weighted FIM term  
            cost += X_err.T@Q@X_err + U[:,k].T@R@U[:,k] + self.lambda_fim@cs.diag(cs.inv(F))
            
            # Dynamics constraint: next state equals current state plus discrete dynamics
            X_next = f(X[:,k], U[:,k])
            g.append(X[:,k+1] - X_next)

            # Update PHI
            PHI += dt*(Jx_k(X[:,k],U[:,k],params)@PHI + Jtht_k(X[:,k],U[:,k],params))
            
        # Terminal cost on the final state
        X_err = self.error(X[:,H], X_cmd) 
        cost += X_err.T@Q@X_err
        
        # Assemble decision vector
        # First: states, shaped as (nx*(H+1),1)
        # Second: controls, shaped as (nu*H, 1)
        opt_vars = cs.vertcat(cs.reshape(X, -1, 1), cs.reshape(U, -1, 1))
        g_concat = cs.vertcat(*g)
        
        # Define NLP problem
        nlp_problem = {
            'f': cost, # cost function
            'x': opt_vars, # decision variables
            'g': g_concat, # constraints
            'p': X0
        }
        
        # Create NLP solver instance using IPOPT
        opts = {'ipopt.print_level': 0, 'print_time': 0}
        solver = cs.nlpsol('solver', 'ipopt', nlp_problem, opts)
        
        return solver
    
    # --- 

    def error(self, x, x_cmd):
        '''
            Compute the error between the current and commanded states
        '''

        # extract states
        p = cs.vertcat(x[0], x[1])
        q = cs.vertcat(x[2], x[3])
        V = cs.vertcat(x[4], x[5], x[6])

        # extract reference states
        pr = cs.vertcat(x_cmd[0], x_cmd[1])
        qr = cs.vertcat(x_cmd[2], x_cmd[3])
        Vr = cs.vertcat(x_cmd[4], x_cmd[5], x_cmd[6])    

        # compute errors
        ep = pr - p
        eq = 1 - (q.T@qr)**2
        eV = Vr - V
        e = cs.vertcat(ep, eq, eV)

        return e
    
    # --- 

    def solve(self, x):
        '''
        Solve the nonlinear OCP for one timestep

        Parameters
        ----------
        x : 7x1 state vector
            x[0] : x position in the inertial frame (m)
            x[1] : y position in the inertial frame (m)
            x[2] : qw real quaternion component
            x[3] : qz imaginiary quaternion component
            x[4] : x velocity in the body frame (m/s)
            x[5] : y velocity in the vody frame (m/s)
            x[6] : z axis rotational velocity (rad/s)
        
        Returns
        -------
        u : 8x1 control vector
            thruster actuation (fraction of max thrust)
        '''
        # Extract parameters
        solver = self.solver
        H = self.H
        lbx = self.lbx
        ubx = self.ubx
        lbg = self.lbg
        ubg = self.ubg
        
        # Solve the OCP for the current state
        sol = solver(p=x, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
        sol_opt = sol['x'].full().flatten()

        # Extract the first control action from the sequence
        u_opt = sol_opt[self.nx*(H+1):self.nx*(H+1)+self.nu*H]
        u = u_opt[0:self.nu]
        
        return u
    
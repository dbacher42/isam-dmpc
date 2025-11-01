import numpy as np
import casadi as cs
import control as ct

class FloatbotLQR():
    def __init__(self, model, x_cmd, dt, Q=np.eye(6), R=np.eye(4), u_cmd=np.zeros(4), max_thrust=np.inf):
        '''
        Class defining the MPC controller for a floatbot

        Parameters
        ----------
        model : floatbot kinematics and dynamics model
        x_cmd : commanded states
        dt : controller timestep
        Q : weight matrix penalizing state errors
            optional, the default is np.eye(7)
        R : weight matrix penalizing control effort
            optional, the default is np.eye(4)
        u_cmd : commanded inputs
            optional, the default is np.zeros(4)
        '''
        # Parameters
        self.model = model
        self.x_cmd = x_cmd
        self.dt = dt
        self.Q = Q
        self.R = R
        self.u_cmd = u_cmd
        self.u_max = max_thrust
        
        # Vector sizes
        self.nx = 7 # number of states
        self.nu = 4 # number of control inputs
        
        # Linearize the model
        self.A, self.B = self.linearize(x_cmd, u_cmd)
        self.A = np.array(self.A)
        self.B = np.array(self.B)
        
        # Remove uncontrollable state (qw)
        self.Ac = np.delete(self.A, 2, 0) # delete row at index 2
        self.Ac = np.delete(self.Ac, 2, 1) # delete column at index 2
        self.Bc = np.delete(self.B, 2, 0) # delete row at index 2
        
        # Discretize system model
        self.sysc = ct.ss(self.Ac, self.Bc, np.eye(self.nx-1), 0)
        self.sysd = ct.c2d(self.sysc, self.dt)
        
        # Compute LQR gains
        self.K, self.S, self.E = ct.dlqr(self.sysd, self.Q, self.R)
        
    def linearize(self, x_cmd, u_cmd):
        '''
        linearize the system about the commanded point
        '''
        # Extract parameters
        xdot = self.model.xdot
        
        # Symbolic vectors
        x = cs.SX.sym('x', self.nx)
        u = cs.SX.sym('u', self.nu)
        f = xdot(x, u)
        
        # Compute Jacobians
        Jx = cs.jacobian(f, x)
        Ju = cs.jacobian(f, u)
        
        # Evaluate at the commanded point
        A = cs.Function('A', [x, u], [Jx])
        B = cs.Function('B', [x, u], [Ju])
        Ae = A(x_cmd, u_cmd)
        Be = B(x_cmd, u_cmd)
        
        return Ae, Be
    
    def error(self, x):
        '''
        Compute the error between the current and commanded states
        '''
        r = np.vstack([x[0], x[1]])
        q = np.vstack([x[2], -x[3]])
        v = np.vstack([x[4], x[5], x[6]])
        
        r_cmd = cs.vertcat(self.x_cmd[0], self.x_cmd[1])
        q_cmd = cs.vertcat(self.x_cmd[3], self.x_cmd[2])
        v_cmd = cs.vertcat(self.x_cmd[4], self.x_cmd[5], self.x_cmd[6])
        
        re = r - r_cmd
        qe = -q_cmd.T@q
        ve = v - v_cmd
        
        return np.vstack([re, qe, ve])
    
    def solve(self, x):
        '''
        Solve the linear control problem for one time step
        '''
        # Extract parameters
        K = self.K

        # Compute control inputs
        x_err = self.error(x)
        u = -K@x_err
        
        return np.clip(u, -self.u_max, self.u_max)
        
if __name__ == "__main__":
    from model import FloatbotModel
    
    # Floatbot parameters
    mass = 16.8
    inertia = .1594
    max_thrust = 1.5
    moment_arm = .12
    cg = np.array([.068, 0])
    
    # States
    rx = 0
    ry = 0
    tht = np.pi/2
    qw = np.cos(tht/2)
    qz = np.sin(tht/2)
    vx = 0
    vy = 0
    wz = 0
    x = np.array([rx, ry, qw, qz, vx, vy, wz])
    
    # Commanded States
    rx_cmd = 0
    ry_cmd = 0
    tht_cmd = 0
    qw_cmd = np.cos(tht_cmd/2)
    qz_cmd = np.sin(tht_cmd/2)
    vx_cmd = 0
    vy_cmd = 0
    wz_cmd = 0
    x_cmd = np.array([rx_cmd, ry_cmd, qw_cmd, qz_cmd, vx_cmd, vy_cmd, wz_cmd])
    
    # LQR parameters
    dt = .1
    Q = np.diag([5e1,5e1,8e3,1e1,1e1,1e1])
    R = 1e-1*np.eye(4)
    
    floatbot_model = FloatbotModel(mass, inertia, max_thrust, moment_arm, cg)
    floatbot_lqr = FloatbotLQR(floatbot_model, x_cmd, dt)
    u = floatbot_lqr.solve(x)
    print(u)
        
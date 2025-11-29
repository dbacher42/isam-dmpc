import numpy as np
import casadi as cs

class FloatbotModel():
    def __init__(self, mass, inertia, moment_arm, cg=np.zeros(2), ri=[np.zeros(2)]):
        '''
        Class defining 2D floatbot kinematic and dynamic model

        Parameters
        ----------
        mass : floatbot mass (kg)
        inertia : inertia about the rotation axis (kg*m**2) 
        max_thrust : maximum thrust produced by thrusters (N)
        moment_arm : tangential distance to thrusters from centroid (m)
        cg : offset of the cg from the centroid, defined in the body frame (m) 
            optional, defaults to [0, 0]
        ri : actuator centroid offset from structure's centroid, defined in the body frame (m) 
            optional, defaults to [0, 0]
        '''
        # Parameters
        self.mass = mass # kg
        self.inertia = inertia # kg*m**2
        self.moment_arm = moment_arm # m
        self.cg = cg # m
        self.ri = ri # m
        self.n = len(self.ri)
        
    def xdot(self, x, u):
        '''
        Computes the floatbot state derivatives

        Parameters
        ----------
        x : 7x1 state vector
            x[0] : x position in the inertial frame (m)
            x[1] : y position in the inertial frame (m)
            x[2] : qw real quaternion component
            x[3] : qz imaginiary quaternion component 
            (note: normalize quaternion to prevent numerical errors)
            x[4] : x velocity in the body frame (m/s)
            x[5] : y velocity in the vody frame (m/s)
            x[6] : z axis rotational velocity (rad/s)
        u : 4*nx1 control vector
            thruster force (N)

        Returns
        -------
        xd : 7x1 casadi.MX object containing the state derivatives
        '''
        
        def rot(q):
            '''
            2x2 z axis rotation matrix from quaternion
            '''
            qw = q[0]
            qz = q[1]
            
            return cs.vertcat(
                cs.horzcat(qw**2-qz**2, -2*qw*qz),
                cs.horzcat(2*qw*qz, qw**2-qz**2)
            )
        
        # Extract parameters
        m = self.mass
        Izz = self.inertia
        rT = self.moment_arm
        cg_B = cs.vertcat(self.cg[0], self.cg[1])
        
        # Extract states
        q = cs.vertcat(x[2],x[3])
        R_BI = rot(q)
        cg_I = R_BI@cg_B
        cx = cg_I[0]
        cy = cg_I[1]
        vx = x[4]
        vy = x[5]
        wz = x[6]
        v = cs.vertcat(vx, vy)
        
        # Kinematics
        rdot = v
        Omg = cs.vertcat(
            cs.horzcat(0, -wz),
            cs.horzcat(wz, 0)
        )
        qdot = 1/2*Omg@q
        
        # Compute forces from input vector
        f_I = []
        t = []
        for i in range(self.n):
            f_B = cs.vertcat(
                u[i*4+0] + u[i*4+2],
                u[i*4+1] + u[i*4+3]
            )
            f_I.append(R_BI@f_B)
            ri = self.ri[i]
            t.append(-rT*(u[i*4+0] + u[i*4+1] - u[i*4+2] - u[i*4+3]) + ri[0]*f_B[1] - ri[1]*f_B[0])

        F = cs.vertcat(
            sum(f_I),
            sum(t)
        )
        
        # Dynamics
        M = cs.vertcat(
            cs.horzcat(m, 0, -m*cy),
            cs.horzcat(0, m, m*cx),
            cs.horzcat(-m*cy, m*cx, Izz)
        )
        Minv = cs.inv(M)
        C = cs.vertcat(
            -m*cx*wz**2,
            -m*cy*wz**2,
            0
        )
        Vdot = Minv@(F-C)
        
        return cs.vertcat(rdot, qdot, Vdot)
    
if __name__ == "__main__":
    
    # Floatbot Parameters
    mass = 16.8
    inertia = .1594
    max_thrust = 1.5
    moment_arm = .12
    cg = np.array([0,0])
    floatbot_positions = [np.vstack([-.15,0]), np.vstack([.15,0])]
    
    # States
    rx = 0
    ry = 0
    tht = 0
    qw = np.cos(tht/2)
    qz = np.sin(tht/2)
    vx = 0
    vy = 0
    omg = 1
    x = np.array([rx, ry, qw, qz, vx, vy, omg])
    
    # Control inputs
    u = np.array([0, 0, 0, 0, 1, 0, 0, 0])
    
    superstructure_model = FloatbotModel(mass, inertia, moment_arm, cg, floatbot_positions)
    
    # Compute the state derivatives for the initial conditions
    for i in range(10):
        xdot = superstructure_model.xdot(x, u)
        x = x + .1*xdot
        
        # Normalize quaternion
        qw = x[2]
        qz = x[3]
        q_mag = np.sqrt(qw**2+qz**2)
        x[2] = qw/q_mag
        x[3] = qz/q_mag
        
        print(x)
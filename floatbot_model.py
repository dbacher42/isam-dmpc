import numpy as np
import casadi as cs

class FloatbotModel():
    def __init__(self, mass, inertia, max_thrust, moment_arm, cg=np.zeros(2)):
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
        '''
        # Parameters
        self.mass = mass # kg
        self.inertia = inertia # kg*m**2
        self.max_thrust = max_thrust # N
        self.moment_arm = moment_arm # m
        self.cg = cg # m
        
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
        u : 8x1 control vector
            thruster actuation (fraction of max thrust)

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
        Tmax = self.max_thrust
        rT = self.moment_arm
        cg_B = self.cg
        
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
        f_B = cs.vertcat(
            Tmax*(u[0] + u[2]),
            Tmax*(u[1] + u[3])
        )
        f_I = R_BI@f_B
        t = -rT*Tmax*(u[0] + u[1] - u[2] - u[3])
        F = cs.vertcat(f_I, t)
        
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
    cg = np.array([.068,0])
    
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
    u = np.array([0, 0, 0, 0, 0, 0, 0, 0])
    
    floatbot_model = FloatbotModel(mass, inertia, max_thrust, moment_arm, cg)
    
    # Compute the state derivatives for the initial conditions
    for i in range(10):
        xdot = floatbot_model.xdot(x, u)
        x = x + .1*xdot
        
        # Normalize quaternion
        qw = x[2]
        qz = x[3]
        q_mag = np.sqrt(qw**2+qz**2)
        x[2] = qw/q_mag
        x[3] = qz/q_mag
        
        print(x)
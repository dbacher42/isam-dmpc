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
        
    def error(self, x, x_cmd):
        '''
        Compute the error between the current and commanded states
        '''
        r = cs.vertcat(x[0], x[1])
        q = cs.vertcat(x[2], x[3])
        v = cs.vertcat(x[4], x[5], x[6])
        
        r_cmd = cs.vertcat(x_cmd[0], x_cmd[1])
        q_cmd = cs.vertcat(x_cmd[2], x_cmd[3])
        v_cmd = cs.vertcat(x_cmd[4], x_cmd[5], x_cmd[6])
        
        re = r - r_cmd
        qe = 1-(q.T@q_cmd)**2
        ve = v - v_cmd
        
        return cs.vertcat(re, qe, ve)
        
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
        cx = self.cg[0] 
        cy = self.cg[1]
        
        # Extract states
        q = cs.vertcat(x[2],x[3])
        R_BI = rot(q)
        vx = x[4]
        vy = x[5]
        wz = x[6]
        v = cs.vertcat(vx, vy)
        V = cs.vertcat(v, wz)
        
        # Kinematics
        rdot = R_BI@v
        Omg = cs.vertcat(
            cs.horzcat(0, -wz),
            cs.horzcat(wz, 0)
        )
        qdot = 1/2*Omg@q
        
        # Compute body frame forces from input vector
        F = cs.vertcat(
            Tmax*(u[0] - u[1] - u[4] + u[5]),
            Tmax*(u[2] - u[3] - u[6] + u[7]),
            -rT*Tmax*(u[0] - u[1] + u[2] - u[3] + u[4] - u[5] + u[6] - u[7])
        )
        
        # Dynamics
        M = cs.vertcat(
            cs.horzcat(m, 0, -m*cy),
            cs.horzcat(0, m, m*cx),
            cs.horzcat(-m*cy, m*cy, Izz)
        )
        Minv = cs.inv(M)
        C = cs.vertcat(
            cs.horzcat(0, -m*wz, -m*cx*wz),
            cs.horzcat(m*wz, 0, -m*cy*wz),
            cs.horzcat(0, 0, cx*vx+cy*vy)
        )
        Vdot = Minv@(F-C@V)
        
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
    omg = -1
    x = np.array([rx, ry, qw, qz, vx, vy, omg])
    
    # Control inputs
    u = np.array([0, 0, 0, 0, 0, 0, 0, 0])
    
    floatbot_model = FloatbotModel(mass, inertia, max_thrust, moment_arm, cg)
    
    # Compute the state derivatives for the initial conditions
    for i in range(10):
        xdot = floatbot_model.xdot(x, u)
        x = x + .1*xdot
        print(x)
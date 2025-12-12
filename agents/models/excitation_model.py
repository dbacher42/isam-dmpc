import numpy as np
import casadi as cs

class excitation_model():

    def __init__(self, mass, inertia, moment_arm, cg=np.zeros(2)):
        '''
        Class defining 2D floatbot kinematic and dynamic model

        --- FOR EXCITATION TRAJECTORY GENERATION --- 
           
        Incorporates a Fisher Information matrix w/ the following 
        estimation parameters of interest:

            tht : 4x1 parameter vector
                tht[0] : mass (kg)                              - of the agent AND structure being manipulated   *** CONFIRM *** 
                tht[1] : x cg offset in the body frame (m)      - from agent to structure
                tht[2] : y cg offset in the body frame (m)      - from agent to structure
                tht[3] : z axis moment of inertia (kg*m**2)     - of the agent AND structure being manipulated
        
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
        self.moment_arm = moment_arm # m
        self.cg = cg # m
    
    # ---

    def xdot(self, x, u, tht):
        '''
        Computes the floatbot state derivatives - WITH information parameters. 

        Note: The agent is initialized with its starting set of these parameters, 
              which are the same that it's trying to estimate. The sim keeps a copy of this estimate 
              which is used in the dynamics here, but the original values are not yet changed. 

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
        
        def mixer(u, R):
            """ 
                Mix thruster position and orientation to agent body frame 
            """ 

            f_B = cs.vertcat(
                u[0]+u[2],
                u[1]+u[3]
            )
            f_I = R@f_B
            tau = self.moment_arm*(-u[0]-u[1]+u[2]+u[3])
            F = cs.vertcat(f_I, tau)

            return F

        def rot(qw, qz):
            """
                2x2 z axis rotation matrix from quaternion
            """
            R = cs.vertcat(
                cs.horzcat(qw**2-qz**2, -2*qw*qz),
                cs.horzcat(2*qw*qz, qw**2-qz**2)
                )
            
            return R

         # extract states
        qw = x[2]
        qz = x[3]
        vx = x[4]
        vy = x[5]
        wz = x[6]

        # extract parameters
        m = tht[0]
        rhox = tht[1]
        rhoy = tht[2]
        Jzz = tht[3]

        # rotation matrix
        R = rot(qw, qz)

        # thruster allocation
        F = mixer(u, R)

        # dynamics
        rho_B = cs.vertcat(rhox, rhoy)
        rho_I = R*rho_B
        M = cs.vertcat(
            cs.horzcat(m, 0, -m*rho_I[1]),
            cs.horzcat(0, m, m*rho_I[0]),
            cs.horzcat(-m*rho_I[1], m*rho_I[0], Jzz)
        )
        C = cs.vertcat(
            -m*rho_I[0]*wz**2,
            -m*rho_I[1]*wz**2,
            0
        )
        Vd = cs.inv(M)@(F-C)

        # kinematics
        v = cs.vertcat(vx, vy)
        pd = v
        q = cs.vertcat(qw, qz)
        Omg = cs.vertcat(
            cs.horzcat(0, -wz),
            cs.horzcat(wz, 0)
        )
        qd = 1/2*Omg@q

        xdot = cs.vertcat(pd, qd, Vd)

        return xdot

# mujoco_playground.py 

""" More for playing w/ MJC-related DMPC sim tasks. """

# -------------------------------------------------------------------------------------------------

import numpy as np 
import matplotlib.pyplot as plt

from DMPC_Sim import *
from worlds import *
from agents import *

# -------------------------------------------------------------------------------------------------

# --- --- --- --- ---    SIM EXEC    --- --- --- --- --- 

dt  = 0.1
sim = DMPC_Sim(dt)


# --- --- --- --- --- SUPERSTRUCTURE --- --- --- --- --- 

blocks, connections = build_ring(size=7)
sim.build_superstructure(blocks, connections)


# --- --- --- --- --- AGENT SETUP --- --- --- --- --- 

# Cardinal directions
N , S , E , W  = np.pi/2, 3*np.pi/2,         0,     np.pi
NE, NW, SW, SE = np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4

def angle_to_quat(angle):
    """Convert angle to [qw, qz]"""
    return [np.cos(angle/2), np.sin(angle/2)]

# Floatbot model (common)
mass       = 53.76
inertia    = 33.11
moment_arm = 0.12
cg         = np.array([0, 0])
model      = FloatbotModel(mass, inertia, moment_arm, cg)

# Controller weights (currently common)
H = 20
Q = np.diag([5e1, 5e1, 1e3, 1e1, 1e1, 1e1])
R = 1e-1 * np.eye(4)

x0s   = [ 
            np.array([-10, -10, *angle_to_quat(E), 0, 0, 0]), # mpc
            np.array([-10,  10, *angle_to_quat(E), 0, 0, 0]),
            np.array([ 10, -10, *angle_to_quat(W), 0, 0, 0]),
            np.array([ 10,  10, *angle_to_quat(W), 0, 0, 0]),

            np.array([-10,   5, *angle_to_quat(E), 0, 0, 0]), # lqr
            np.array([ -5, -10, *angle_to_quat(N), 0, 0, 0]),
            np.array([ 10,  -5, *angle_to_quat(W), 0, 0, 0]),
            np.array([  5,  10, *angle_to_quat(S), 0, 0, 0]),
        ]

xcmds = [ 
            np.array([ 10,   0, *angle_to_quat(N), 0, 0, 0]),
            np.array([  0, -10, *angle_to_quat(N), 0, 0, 0]),
            np.array([  0,  10, *angle_to_quat(S), 0, 0, 0]),
            np.array([-10,   0, *angle_to_quat(S), 0, 0, 0]),

            np.array([  5, -10, *angle_to_quat(S), 0, 0, 0]),
            np.array([ 10,   5, *angle_to_quat(E), 0, 0, 0]),
            np.array([ -5,  10, *angle_to_quat(N), 0, 0, 0]),
            np.array([-10,  -5, *angle_to_quat(S), 0, 0, 0]),
        ]

control = [1, 1, 1, 1, 0, 0, 0, 0]  # 1: MPC, 0: LQR


# --- --- --- --- --- BUILD AGENTS --- --- --- --- --- 

for i in range(len(x0s)):
    
    x0      = x0s[i]
    x_cmd   = xcmds[i]
    ctrl_id = control[i]
    
    if ctrl_id == 0:  
        ctrl_type   = 'LQR'
        ctrl_params = {'x_cmd': x_cmd, 'Q': Q, 'R': R}
    else:            
        ctrl_type   = 'MPC'
        ctrl_params = {'x_cmd': x_cmd, 'H': H, 'Q': Q, 'R': R}
    
    name  = f"Agent_{i+1}_{ctrl_type}"
    sim.build_agent(name, model, x0, ctrl_type, **ctrl_params)
    print(f" Built {name}. Start: {x0[:4]} → Target: {x_cmd[:4]}")


# --- --- --- --- ---      RUN     --- --- --- --- --- 

sim.finalize()
sim._view_mjc()
import os
import mujoco
from mujoco import viewer
import numpy as np
from model import FloatbotModel
from mpc import FloatbotMPC

# Floatbot parameters
mass = 36.96
inertia = 1.956
max_thrust = 1.5
moment_arm = .12
floatbot_positions = [np.vstack([-.6,0]), np.vstack([.6,0]), np.vstack([0,.6])]

# States
rx = 1
ry = 0
tht = 0
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
    
# MPC parameters
dt = .1
H = 20
Q = np.diag([5e1,5e1,8e3,1e1,1e1,1e1])
R = 1e-1*np.eye(4*3)
    
structure_model = FloatbotModel(mass, inertia, moment_arm, ri=floatbot_positions)
floatbot_mpc = FloatbotMPC(structure_model, x_cmd, dt, H, Q, R, max_thrust)

current_dir = os.path.dirname(__file__)
xml_path = os.path.join(current_dir, 'world.xml')

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# Set initial conditions
data.qpos = [rx, ry, tht]
data.qvel = [vx, vy, wz]

def mpc_callback(model, data):

    x = [data.qpos[0], data.qpos[1], np.cos(data.qpos[2]/2), np.sin(data.qpos[2]/2), data.qvel[0], data.qvel[1], data.qvel[2]]
    u = floatbot_mpc.solve(x)

    data.ctrl = u

last_control_time = 0

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():

        if data.time >= last_control_time + dt:
            mpc_callback(model, data)
            last_control_time = data.time


        mujoco.mj_step(model, data)
        viewer.sync()
# scenarios/ring_formation.py

"""
Multi-agent formation control scenario with ring superstructure.

8 agents (4 MPC + 4 LQR) positioned around a ring structure,
with formation control objectives.
"""

import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from DMPC_Sim import *
from worlds import *
from agents import *

def angle_to_quat(angle):
    """Convert angle to [qw, qz] quaternion representation."""
    return [np.cos(angle/2), np.sin(angle/2)]

def run_ring_formation_scenario(
    dt=0.1, 
    sim_time=30.0, 
    mpc_horizon=10, 
    render=True, 
    real_time=True,
    num_agents=8
):
    """
    Run the ring formation scenario.
    
    Parameters
    ----------
    dt : float
        Simulation timestep (default: 0.1s)
    sim_time : float  
        Total simulation time (default: 10s)
    mpc_horizon : int
        MPC prediction horizon (default: 10, reduced from 20 for speed)
    render : bool
        Show real-time MuJoCo visualization (default: True)
    real_time : bool
        Run at real-time speed vs as-fast-as-possible (default: True)
    num_agents : int
        Number of agents to create (default: 8)
    """
    
    print(f"=== Ring Formation Scenario ===")
    print(f"Agents: {num_agents}, Horizon: {mpc_horizon}, Time: {sim_time}s")
    

    # --- SIMULATION SETUP ---
    sim = DMPC_Sim(dt)
    

    # --- SUPERSTRUCTURE ---

    # Build ring 
    blocks, connections = build_ring(size=7)
    sim.build_superstructure(blocks, connections)
    sim.compile()
    
    # Attachment points 
    positions = sim.get_block_positions()

    # --- AGENT CONFIGURATION ---
    
    # Cardinal directions
    N, S, E, W = np.pi/2, 3*np.pi/2, 0, np.pi
    NE, NW, SW, SE = np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4
    
    # Floatbot model parameters
    mass = 18.48
    inertia = .1462 
    moment_arm = 0.12
    cg = np.array([0, 0])
    model = FloatbotModel(mass, inertia, moment_arm, cg)
    
    # Controller weights
    Q = np.diag([5e1, 5e1, 1e3, 1e2, 1e2, 1e1])
    R = 1e-1 * np.eye(4)
    
    # Initial positions and targets
    x0s = [
        np.array([-10, -10, *angle_to_quat(E), 0, 0, 0]),  # MPC agents
        np.array([-10,  10, *angle_to_quat(E), 0, 0, 0]),
        np.array([ 10, -10, *angle_to_quat(W), 0, 0, 0]),
        np.array([ 10,  10, *angle_to_quat(W), 0, 0, 0]),
        np.array([-10,   5, *angle_to_quat(E), 0, 0, 0]),  # LQR agents
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
    
    # Controller types (1: MPC, 0: LQR)
    control = [1, 1, 1, 1, 1, 1, 1, 1]
    

    # --- BUILD AGENTS ---
    
    for i in range(min(num_agents, len(x0s))):
        x0 = x0s[i]
        x_cmd = xcmds[i]
        ctrl_id = control[i]
        
        if ctrl_id == 0:
            ctrl_type = 'LQR'
            ctrl_params = {'x_cmd': x_cmd, 'Q': Q, 'R': R}
        else:
            ctrl_type = 'MPC'
            ctrl_params = {'x_cmd': x_cmd, 'H': mpc_horizon, 'Q': Q, 'R': R, 'max_thrust': 1.5}
        
        name = f"Agent_{i+1}_{ctrl_type}"
        sim.build_agent(name, model, x0, ctrl_type, **ctrl_params)
        print(f"  Built {name}: {x0[:2]} → {x_cmd[:2]}")
    

    # --- RUN SIMULATION ---
    
    sim.finalize(True)
    sim.run(sim_time, render=render, real_time=real_time)
    

    # --- RESULTS ---
    
    print("\\n=== Final States ===")
    for name, agent_data in sim.Oracle.data.items():
        start_pos = agent_data['state'][0][:2]
        final_pos = agent_data['state'][-1][:2]
        distance = np.linalg.norm(final_pos - start_pos)
        print(f"{name}: {start_pos.round(2)} → {final_pos.round(2)} (moved: {distance:.2f})")
    
    return sim


# --- CONVENIENCE FUNCTIONS ---

def quick_test():
    """Quick test with reduced parameters for development."""
    return run_ring_formation_scenario(
        sim_time=5.0,
        mpc_horizon=5,
        num_agents=2,
        real_time=False
    )

def full_demo():
    """Full demonstration with all agents."""
    return run_ring_formation_scenario(
        sim_time=15.0,
        mpc_horizon=8,
        render=True,
        real_time=True
    )

# --- MAIN ---

if __name__ == "__main__":
    # Default run
    run_ring_formation_scenario()
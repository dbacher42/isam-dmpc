# scenarios/test_docking.py

"""
Test scenario for agent docking to superstructure.

4 MPC agents docked to middle blocks of each side of a 5x5 ring.
Agents start at attachment points below structure blocks.
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

def run_docking_test(
    dt=0.1, 
    sim_time=25.0, 
    mpc_horizon=10, 
    render=True, 
    real_time=True,
    z_offset=-0.5
):
    """
    Run the docking test scenario.
    
    Parameters
    ----------
    dt : float
        Simulation timestep (default: 0.1s)
    sim_time : float  
        Total simulation time (default: 10s)
    mpc_horizon : int
        MPC prediction horizon (default: 10)
    render : bool
        Show real-time MuJoCo visualization (default: True)
    real_time : bool
        Run at real-time speed vs as-fast-as-possible (default: True)
    z_offset : float
        Z offset for agent attachment below blocks (default: -0.5)
    """
    
    print(f"=== Docking Test Scenario ===")
    print(f"4 MPC agents on 5x5 ring, Horizon: {mpc_horizon}, Time: {sim_time}s")
    

    # --- SIMULATION SETUP ---
    sim = DMPC_Sim(dt)
    

    # --- SUPERSTRUCTURE ---

    # Build 5x5 ring 
    blocks, connections = build_ring(size=5)
    sim.build_superstructure(blocks, connections)
    sim.compile()
    
    # Get attachment points
    attach_pts = sim.get_attachment_points()#z_offset=z_offset)
    
    # Debug: print available blocks
    print(f"\nAvailable blocks: {list(attach_pts.keys())}")
    

    # --- AGENT CONFIGURATION ---
    
    # Middle blocks of each side for 5x5 ring:
    # Top side:    block_2_4 (middle of row y=4)
    # Bottom side: block_2_0 (middle of row y=0)  
    # Right side:  block_4_2 (middle of col x=4)
    # Left side:   block_0_2 (middle of col x=0)
    
    dock_blocks = ['block_2_4', 'block_2_0', 'block_4_2', 'block_0_2']
    
    # Cardinal directions (heading facing inward toward ring center)
    headings = [0] * 4 # [np.pi/2, np.pi/2, np.pi/2, np.pi/2]  # all N
    
    # Floatbot model parameters
    mass = 18.48
    inertia = .1462 
    moment_arm = 0.12
    cg = np.array([0, 0])
    model = FloatbotModel(mass, inertia, moment_arm, cg)
    
    # Controller weights
    Q = np.diag([5e1, 5e1, 1e3, 1e2, 1e2, 1e1])
    R = 1e-1 * np.eye(4)
    

    # --- BUILD AGENTS ---
    
    # When docked, all agents must have CONSISTENT targets (same delta movement)
    # since they move as one rigid body
    target_delta_y = 5.0  # Move structure +5 in Y (North)
    
    for i, block_id in enumerate(dock_blocks):
        # Get attachment point
        pos = attach_pts[block_id]
        heading = headings[i]
        
        # Initial state: position from attachment, heading, zero velocity
        x0 = np.array([pos[0], pos[1], *angle_to_quat(heading), 0, 0, 0])
        
        # Command = move by same delta for all agents (coordinated)
        # Each agent targets its own position shifted by the same amount
        x_cmd = np.array([pos[0], pos[1] + target_delta_y, *angle_to_quat(heading), 0, 0, 0])
        
        # Build MPC agent with docking
        name = f"Agent_{i+1}"
        ctrl_params = {'x_cmd': x_cmd, 'H': mpc_horizon, 'Q': Q, 'R': R, 'max_thrust': 1.5}
        
        # Use dock_block parameter for proper positioning
        agent = sim.build_agent(name, model, x0, 'MPC', target_z=pos[2], dock_block=block_id, **ctrl_params)
        print(f"  Built {name} at {block_id}: pos=({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")
    

    # --- RUN SIMULATION ---
    
    sim.finalize()
    sim.run(sim_time, render=render, real_time=real_time)
    

    # --- RESULTS ---
    
    print("\n=== Final States ===")
    for name, agent_data in sim.Oracle.data.items():
        start_pos = agent_data['state'][0][:2]
        final_pos = agent_data['state'][-2][:2]
        distance = np.linalg.norm(final_pos - start_pos)
        print(f"{name}: {start_pos.round(2)} → {final_pos.round(2)} (moved: {distance:.2f})")
    
    
    sim.plot_trajectories()
    sim.plot_states()
    sim.plot_controls()


    return sim


# --- MAIN ---

if __name__ == "__main__":
    run_docking_test()

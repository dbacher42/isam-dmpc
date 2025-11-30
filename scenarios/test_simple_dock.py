# scenarios/test_simple_dock.py

"""
Minimal test: single block structure, single agent docked, move north.
"""

import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from DMPC_Sim import *
from superstructure import Unit_Block, Connection, Superstructure_2D
from agents import *


def angle_to_quat(angle):
    """Convert angle to [qw, qz] quaternion representation."""
    return [np.cos(angle/2), np.sin(angle/2)]


def run_test():
    print("=== Simple Dock Test ===")
    print("1 block, 1 agent, move 10 units north\n")
    
    dt = 0.1
    sim_time = 30.0
    
    # --- SIMULATION ---
    sim = DMPC_Sim(dt)
    
    # --- SINGLE BLOCK STRUCTURE ---
    blocks = [Unit_Block("block_0", size=1.0)]
    connections = []  # No connections needed for single block
    
    # Build structure
    sim.build_superstructure(blocks, connections)
    sim.compile()
    
    # Get block position
    attach = sim.get_attachment_points('block_0')
    print(f"Unit Structure Position: {attach[0:2]}")
    
    # Target Z for docking (bottom of block)
    target_z = attach[2]                            # target is lower face of block 
    print(f"Target Z for docking: {target_z}")
    
    # --- AGENT ---
    mass = 18.48
    inertia = 0.1462
    moment_arm = 0.12
    cg = np.array([0, 0])
    model = FloatbotModel(mass, inertia, moment_arm, cg)
    
    # Heading = North (pi/2)
    heading = 0#np.pi / 2
    
    # Initial state at block position
    x0 = np.array([attach[0], attach[1], *angle_to_quat(heading), 0, 0, 0])
    
    # Target = 10 units north
    x_cmd = np.array([attach[0], attach[1] + 5.0, *angle_to_quat(heading), 0, 0, 0])
    
    print(f"Initial state: pos=({x0[0]:.1f}, {x0[1]:.1f}), heading={heading:.2f}")
    print(f"Target state:  pos=({x_cmd[0]:.1f}, {x_cmd[1]:.1f})")
    
    # Controller
    Q = np.diag([5e1, 5e1, 1e3, 1e2, 1e2, 1e1])
    R = 1e-1 * np.eye(4)
    ctrl_params = {'x_cmd': x_cmd, 'H': 10, 'Q': Q, 'R': R, 'max_thrust': 1.5}
    
    # Build agent with docking
    agent = sim.build_agent("Agent_1", model, x0, 'MPC', target_z=target_z, dock_block="block_0", **ctrl_params)
    print(f"Agent z_offset: {agent.z_offset}")
    
    # --- RUN ---
    sim.finalize()
    
    # Print some debug info
    print(f"\nModel has {sim.model.neq} equality constraints")
    print(f"Model has {sim.model.njnt} joints")
    print(f"Model has {sim.model.nbody} bodies")
    
    # Check structure mass
    struct_body_id = sim.model.body("block_0").id
    print(f"Structure body mass (from subtree): {sim.model.body_subtreemass[struct_body_id]:.2f} kg")
    
    # Check agent mass  
    agent_body_id = sim.model.body("Agent_1_floatbot").id
    print(f"Agent body mass (from subtree): {sim.model.body_subtreemass[agent_body_id]:.2f} kg")
    
    print("\n--- Running simulation ---")
    sim.run(sim_time, render=True, real_time=True)
    
    # --- RESULTS ---
    print("\n=== Results ===")
    agent_data = sim.Oracle.data['Agent_1']
    
    start_pos = agent_data['state'][0][:2]
    final_pos = agent_data['state'][-1][:2]
    distance = np.linalg.norm(final_pos - start_pos)
    
    print(f"Start: {start_pos}")
    print(f"Final: {final_pos}")
    print(f"Distance moved: {distance:.2f}")
    
    # Check control history
    ctrl_hist = np.array(agent_data['control'])
    print(f"\nControl history shape: {ctrl_hist.shape}")
    print(f"Mean controls: {ctrl_hist.mean(axis=0)}")
    print(f"Max controls:  {ctrl_hist.max(axis=0)}")
    print(f"Min controls:  {ctrl_hist.min(axis=0)}")
    
    # Plot
    sim.plot_trajectories()
    sim.plot_states()
    sim.plot_controls()

    return sim


if __name__ == "__main__":
    run_test()

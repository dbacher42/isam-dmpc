# scenarios/test_simple_dock.py

"""
    Minimal parameter estimation test: single block structure, single agent docked, move north.
"""

import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from DMPC_Sim import *
from superstructure import Unit_Block, Connection, Superstructure_2D
from agents import *

# -------------------------------------------------------------------------------------------------

def angle_to_quat(angle):
    """Convert angle to [qw, qz] quaternion representation."""
    return [np.cos(angle/2), np.sin(angle/2)]

# --- 

def run_test():
    print("=== Simple Dock Test ===")
    print("1 block, 1 agent, move 10 units north\n")
    
    # ------------------------------- SIM SETUP -------------------------------

    # Sim 
    dt = 0.2
    sim_time = 5.0
    sim = DMPC_Sim(dt)
    
    
    # ------------------------------- SUPERSTRUCTURE -------------------------------

    # Single block test case 
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
    

    # ------------------------------- BUILD AGENT MODEL -------------------------------

    # System model 
    mass = 18.48
    inertia = 0.1462
    moment_arm = 0.12
    cg = np.array([0, 0])
    model = excitation_model(mass, inertia, moment_arm, cg)
    
    # Heading = North (pi/2)
    heading = 0#np.pi / 2
    
    # Initial state is at docked block position 
    x0 = np.array([attach[0], attach[1], *angle_to_quat(heading), 0, 0, 0])
    
    # Target = 5 units north
    x_cmd = np.array([attach[0], attach[1] + 5, *angle_to_quat(heading), 0, 0, 0])
    
    print(f"Initial state: pos=({x0[0]:.1f}, {x0[1]:.1f}), heading={heading:.2f}")
    print(f"Target state:  pos=({x_cmd[0]:.1f}, {x_cmd[1]:.1f})")
    
    # Controller - rebalanced to emphasize excitation
    Q = np.diag([1e1, 1e1, 1e2, 1e1, 1e1, 1e0])  # Reduced position/tracking cost by 5-10x
    R = 1e-1 * np.eye(4)
    lambda_fim = np.array([[100.0], [100.0], [100.0], [100.0]])  # Increased FIM weight 100x
    ctrl_params = {'x_cmd': x_cmd, 'H': 10, 'Q': Q, 'R': R, 'lambda_fim': lambda_fim, 'max_thrust': 1.5}
    

    # ------------------------------- SIM EXECUTION -------------------------------

    # Officially add agent into sim env and dock to structure 
    agent = sim.build_agent("Agent_1", model, x0, 'Excitation_MPC', target_z=target_z, dock_block="block_0", **ctrl_params)
    print(f"Agent z_offset: {agent.z_offset}")
    
    # Compile and finalize all checks 
    sim.finalize()
    
    # Basic checks 
    print(f"\nModel has {sim.model.neq} equality constraints")
    print(f"Model has {sim.model.njnt} joints")
    print(f"Model has {sim.model.nbody} bodies")
    struct_body_id = sim.model.body("block_0").id
    agent_body_id = sim.model.body("Agent_1_floatbot").id
    print(f"Structure body mass (from subtree): {sim.model.body_subtreemass[struct_body_id]:.2f} kg")
    print(f"Agent body mass (from subtree): {sim.model.body_subtreemass[agent_body_id]:.2f} kg")
    
    # Run 
    print("\n--- Running simulation ---")
    sim.run(sim_time, render=True, real_time=True)
    

    # ------------------------------- BASIC RESULTS -------------------------------
    
    print("\n=== Results ===")
    agent_data = sim.Oracle.data['Agent_1']
    
    # x0 / xf 
    start_pos = agent_data['state'][0][:2]
    final_pos = agent_data['state'][-1][:2]
    distance = np.linalg.norm(final_pos - start_pos)
    print(f"Start: {start_pos}")
    print(f"Final: {final_pos}")
    print(f"Distance moved: {distance:.2f}")
    
    # u
    ctrl_hist = np.array(agent_data['control'])
    print(f"\nControl history shape: {ctrl_hist.shape}")
    print(f"Mean controls: {ctrl_hist.mean(axis=0)}")
    print(f"Max controls:  {ctrl_hist.max(axis=0)}")
    print(f"Min controls:  {ctrl_hist.min(axis=0)}")
    
    # Checking for added rotation due to info terms 
    state_hist = np.array(agent_data['state'])
    wz_hist = state_hist[:, 6]  # Angular velocity
    print(f"\nRotation stats:")
    print(f"  wz mean: {wz_hist.mean():.4f} rad/s")
    print(f"  wz std:  {wz_hist.std():.4f} rad/s")
    print(f"  wz max:  {wz_hist.max():.4f} rad/s")
    print(f"  wz min:  {wz_hist.min():.4f} rad/s")
    
    # Check heading change
    heading_start = 2*np.arctan2(state_hist[0, 3], state_hist[0, 2])
    heading_end = 2*np.arctan2(state_hist[-1, 3], state_hist[-1, 2])
    print(f"  Heading change: {np.degrees(heading_end - heading_start):.2f} degrees")
    
    
    # --------------------------- PARAM ESTIMATION AND FULL RES --------------------------

    print("\n=========== Parameter Estimation ===========")
    
    # Get ground truth composite mass properties from simulation
    composite_mass, composite_cg, composite_inertia = sim.get_composite_mass_properties()
    
    print(f"Ground Truth (MuJoCo composite):")
    print(f"  Mass: {composite_mass:.3f} kg")
    print(f"  CG: ({composite_cg[0]:.4f}, {composite_cg[1]:.4f}) m (world frame)")
    print(f"  Inertia: {composite_inertia:.4f} kg*m^2")
    
    # Agent's initial model belief (just after docking, before runtime)
    print(f"\nAgent's Initial Model (solo):")
    print(f"  Mass: {mass:.3f} kg")
    print(f"  Inertia: {inertia:.4f} kg*m^2")
    print(f"  CG: ({cg[0]:.3f}, {cg[1]:.3f}) m")
    
    # Run estimation using Oracle data
    tht_hat = agent.estimate(oracle_data=agent_data, update_model=False)
    
    print(f"\nEstimated Parameters:")
    print(f"  Mass: {tht_hat[0]:.3f} kg")
    print(f"  CG_x: {tht_hat[1]:.4f} m")
    print(f"  CG_y: {tht_hat[2]:.4f} m")
    print(f"  Inertia: {tht_hat[3]:.4f} kg*m^2")
    
    # --- FRAME TRANSFORMATION CHECK ---
    # The estimated CG is in BODY FRAME (rho_x, rho_y at estimation time)
    # The ground truth CG is in WORLD FRAME
    # Need to transform one to match the other
    
    print(f"\n=== Frame Transformation Check ===")
    
    # Get agent's body state during estimation trajectory (use average position/orientation)
    # The estimation uses the full trajectory, so we should use an average or representative pose
    # For simplicity, use the final state
    final_state = state_hist[-1]
    agent_pos_world = final_state[:2]  # [x, y] in world frame
    qw, qz = final_state[2], final_state[3]  # quaternion
    
    print(f"Agent body position (world frame, final): ({agent_pos_world[0]:.4f}, {agent_pos_world[1]:.4f})")
    print(f"Composite CG (world frame): ({composite_cg[0]:.4f}, {composite_cg[1]:.4f})")
    
    # Rotation matrix from body to world frame
    R_body_to_world = np.array([
        [qw**2 - qz**2, -2*qw*qz],
        [2*qw*qz, qw**2 - qz**2]
    ])
    
    # Transform composite CG from world frame to body frame
    # CG_body = R^T @ (CG_world - agent_pos_world)
    cg_world_relative = composite_cg - agent_pos_world  # CG position relative to agent origin
    composite_cg_body = R_body_to_world.T @ cg_world_relative
    
    print(f"Composite CG (body frame): ({composite_cg_body[0]:.4f}, {composite_cg_body[1]:.4f})")
    print(f"Estimated CG (body frame): ({tht_hat[1]:.4f}, {tht_hat[2]:.4f})")
    
    # Now compute errors in the same frame (body frame)
    mass_error = tht_hat[0] - composite_mass
    cg_x_error_body = tht_hat[1] - composite_cg_body[0]
    cg_y_error_body = tht_hat[2] - composite_cg_body[1]
    inertia_error = tht_hat[3] - composite_inertia
    
    print(f"\nEstimation Errors (CG in body frame):")
    print(f"  Mass: {mass_error:.3f} kg ({100*abs(mass_error)/composite_mass:.1f}%)")
    print(f"  CG_x: {cg_x_error_body:.4f} m")
    print(f"  CG_y: {cg_y_error_body:.4f} m")
    if composite_inertia != 0:
        print(f"  Inertia: {inertia_error:.4f} kg*m^2 ({100*abs(inertia_error)/composite_inertia:.1f}%)")
    else:
        print(f"  Inertia: {inertia_error:.4f} kg*m^2 (GT is zero!)")
    
    # Plot
    sim.plot_trajectories()
    sim.plot_states()
    sim.plot_controls()

    return sim


if __name__ == "__main__":
    run_test()

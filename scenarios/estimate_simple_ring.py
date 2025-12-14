# scenarios/estimate_simple_ring.py

"""
    Parameter estimation test: ring structure with single docked agent.
"""

import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from DMPC_Sim import *
from worlds import build_ring
from agents import *

# -------------------------------------------------------------------------------------------------

def angle_to_quat(angle):
    """Convert angle to [qw, qz] quaternion representation."""
    return [np.cos(angle/2), np.sin(angle/2)]

# --- 

def hand_calculate_ring_properties(size=5):
    """
    Hand calculate the expected mass properties for a ring structure.
    
    Ring structure (size=5):
    - 16 blocks arranged in a 5x5 hollow square
    - Each block: 10 kg mass, 1.0 m size, 1.667 kg*m^2 inertia
    
    Returns
    -------
    mass, cg, inertia : float, array, float
        Expected mass (kg), CG position (m), and inertia (kg*m^2)
    """
    # Generate ring positions (same logic as build_ring)
    positions = []
    
    # Top row: (0,N-1) to (N-1,N-1)
    for x in range(size):
        positions.append((x, size-1))
    
    # Right edge: (N-1,N-2) to (N-1,0)
    for y in range(size-2, -1, -1):
        positions.append((size-1, y))
    
    # Bottom row: (N-2,0) to (0,0)
    for x in range(size-2, -1, -1):
        positions.append((x, 0))
    
    # Left edge: (0,1) to (0,N-2)
    for y in range(1, size-1):
        positions.append((0, y))
    
    print(f"\n=== Hand Calculation for Ring (size={size}) ===")
    print(f"Number of blocks: {len(positions)}")
    print(f"Block positions: {positions}")
    
    # Each block properties
    block_mass = 5.0  # kg
    block_size = 1.0   # m
    block_inertia = (1/6) * block_mass * block_size**2  # kg*m^2 for solid cube
    
    print(f"Each block: mass={block_mass} kg, size={block_size} m, inertia={block_inertia:.4f} kg*m^2")
    
    # Total mass
    total_mass = len(positions) * block_mass
    
    # CG (mass-weighted average of positions)
    cg_x = sum(pos[0] for pos in positions) / len(positions)
    cg_y = sum(pos[1] for pos in positions) / len(positions)
    cg = np.array([cg_x, cg_y])
    
    print(f"Structure mass: {total_mass} kg")
    print(f"Structure CG: ({cg_x:.4f}, {cg_y:.4f}) m")
    
    # Total inertia (parallel axis theorem)
    # I_total = sum(I_local_i + m_i * d_i^2)
    total_inertia = 0.0
    for pos in positions:
        d_sq = (pos[0] - cg_x)**2 + (pos[1] - cg_y)**2
        contrib = block_inertia + block_mass * d_sq
        total_inertia += contrib
    
    print(f"Structure inertia: {total_inertia:.4f} kg*m^2")
    print()
    
    return total_mass, cg, total_inertia


def run_single_test(lambda_fim_value, render=False, verbose=True):
    if verbose:
        print("=== Simple Ring Estimation Test ===")
        print("Ring structure, 1 agent docked, parameter estimation\n")
    
    # Hand calculations for reference
    ring_size = 5
    hand_mass, hand_cg, hand_inertia = hand_calculate_ring_properties(ring_size)
    
    # ------------------------------- SIM SETUP -------------------------------

    # Sim 
    dt = 0.2
    sim_time = 10.0
    sim = DMPC_Sim(dt)
    
    
    # ------------------------------- SUPERSTRUCTURE -------------------------------

    # Ring structure test case  
    blocks, connections = build_ring(size=ring_size)
    
    # Build structure
    sim.build_superstructure(blocks, connections)
    sim.compile()
    
    # Get first block position for docking
    first_block = f"block_0_{ring_size-1}"  # Top-left corner: (0, size-1)
    attach = sim.get_attachment_points(first_block)
    print(f"Docking to {first_block}")
    print(f"Attachment position: {attach[0:2]}")
    
    # Target Z for docking (bottom of block)
    target_z = attach[2]
    print(f"Target Z for docking: {target_z}")
    

    # ------------------------------- BUILD AGENT MODEL -------------------------------

    # System model (EXACT same parameters as test_simple_dock)
    mass = 18.48
    inertia = 0.1462
    moment_arm = 0.12
    cg = np.array([0, 0])
    model = excitation_model(mass, inertia, moment_arm, cg)
    
    # Heading = East (0)
    heading = 0
    
    # Initial state is at docked block position 
    x0 = np.array([attach[0], attach[1], *angle_to_quat(heading), 0, 0, 0])
    
    # Target = 5 units north
    x_cmd = np.array([attach[0], attach[1] + 5, *angle_to_quat(heading), 0, 0, 0])
    
    print(f"Initial state: pos=({x0[0]:.1f}, {x0[1]:.1f}), heading={heading:.2f}")
    print(f"Target state:  pos=({x_cmd[0]:.1f}, {x_cmd[1]:.1f})")
    
    # Controller - EXACT same weights as test_simple_dock
    Q = np.diag([1e1, 1e1, 1e2, 1e1, 1e1, 1e0])
    R = 1e-1 * np.eye(4)
    lambda_fim = np.array([[lambda_fim_value], [lambda_fim_value], [lambda_fim_value], [lambda_fim_value]])
    ctrl_params = {'x_cmd': x_cmd, 'H': 10, 'Q': Q, 'R': R, 'lambda_fim': lambda_fim, 'max_thrust': 1.5}
    

    # ------------------------------- SIM EXECUTION -------------------------------

    # Officially add agent into sim env and dock to structure 
    agent = sim.build_agent("Agent_1", model, x0, 'Excitation_MPC', target_z=target_z, 
                           dock_block=first_block, **ctrl_params)
    if verbose:
        print(f"Agent z_offset: {agent.z_offset}")
    
    # Compile and finalize all checks 
    sim.finalize()
    
    # Run simulation
    sim.run(sim_time, render=render, real_time=render)
    
    if verbose:
        print("\n=== Results ===")
    agent_data = sim.Oracle.data['Agent_1']
    
    # x0 / xf 
    start_pos = agent_data['state'][0][:2]
    final_pos = agent_data['state'][-1][:2]
    distance = np.linalg.norm(final_pos - start_pos)
    print(f"Start: {start_pos}")
    print(f"Final: {final_pos}")
    print(f"Distance moved: {distance:.2f}")
    
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
    
    print(f"\nHand Calculation (structure only, for reference):")
    print(f"  {hand_mass:.1f} kg, CG=({hand_cg[0]:.1f}, {hand_cg[1]:.1f}), I={hand_inertia:.1f} kg*m^2")
    
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
    
    mass_pct_error    = 100 * abs(mass_error) / composite_mass
    inertia_pct_error = 100 * abs(inertia_error) / composite_inertia 

    res = [mass_pct_error, cg_x_error_body, cg_y_error_body, inertia_pct_error]

    print(f"\nEstimation Errors (CG in body frame):")
    print(f"  Mass: {mass_error:.3f} kg ({100*abs(mass_error)/composite_mass:.1f}%)")
    print(f"  CG_x: {cg_x_error_body:.4f} m")
    print(f"  CG_y: {cg_y_error_body:.4f} m")
    if composite_inertia != 0:
        print(f"  Inertia: {inertia_error:.4f} kg*m^2 ({100*abs(inertia_error)/composite_inertia:.1f}%)")
    else:
        print(f"  Inertia: {inertia_error:.4f} kg*m^2 (GT is zero!)")
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
    # sim.plot_trajectories()
    # sim.plot_states()
    # sim.plot_controls()

    return sim, res


if __name__ == "__main__":

    import matplotlib.pyplot as plt

     # Lambda sweep: log-spaced from 0.1 to 1000
    lambda_values = np.logspace(-1, 3, 15)
    #lambda_values = np.array([100])
    
    # Collect results
    results = {'lambda': [], 'mass_err': [], 'cg_x_err': [], 'cg_y_err': [], 'inertia_err': []}
    
    for lam in lambda_values:
        print(f"\nTesting lambda_fim = {lam:.2f}")
        sim, res = run_single_test(lam, render=False, verbose=True)
        results['lambda'].append(lam)
        results['mass_err'].append(res[0])
        results['cg_x_err'].append(res[1])
        results['cg_y_err'].append(res[2])
        results['inertia_err'].append(res[3])
    
    import pandas as pd

    df = pd.DataFrame(results)
    # display df nicely
    print("\n=== Summary of Results ===")
    print(df.to_string(index=False))

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0,0].semilogx(results['lambda'], results['mass_err'], 'o-', linewidth=2, markersize=6)
    axes[0,0].set_xlabel('λ_FIM', fontsize=12)
    axes[0,0].set_ylabel('Mass Error (%)', fontsize=12)
    axes[0,0].set_title('Mass Estimation Error', fontsize=13, fontweight='bold')
    axes[0,0].grid(True, alpha=0.3)
    
    axes[0,1].semilogx(results['lambda'], results['inertia_err'], 'o-', linewidth=2, markersize=6, color='C1')
    axes[0,1].set_xlabel('λ_FIM', fontsize=12)
    axes[0,1].set_ylabel('Inertia Error (%)', fontsize=12)
    axes[0,1].set_title('Inertia Estimation Error', fontsize=13, fontweight='bold')
    axes[0,1].grid(True, alpha=0.3)
    
    axes[1,0].semilogx(results['lambda'], results['cg_x_err'], 'o-', linewidth=2, markersize=6, color='C2')
    axes[1,0].set_xlabel('λ_FIM', fontsize=12)
    axes[1,0].set_ylabel('CG_x Error (m)', fontsize=12)
    axes[1,0].set_title('CG X-Position Error', fontsize=13, fontweight='bold')
    axes[1,0].grid(True, alpha=0.3)
    
    axes[1,1].semilogx(results['lambda'], results['cg_y_err'], 'o-', linewidth=2, markersize=6, color='C3')
    axes[1,1].set_xlabel('λ_FIM', fontsize=12)
    axes[1,1].set_ylabel('CG_y Error (m)', fontsize=12)
    axes[1,1].set_title('CG Y-Position Error', fontsize=13, fontweight='bold')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('lambda_sweep_results.png', dpi=150)
    print("\nPlot saved: ring_lambda_sweep_results.png")
    plt.show()


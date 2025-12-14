# scenarios/test_non_uniform_lambdas.py

"""
    Parameter estimation with NON-UNIFORM lambda_fim sweep: 
    - CG weights frozen at 100
    - Mass weight and Inertia weight varied independently on a grid
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from DMPC_Sim import *
from superstructure import Unit_Block, Connection, Superstructure_2D
from agents import *

# -------------------------------------------------------------------------------------------------

def angle_to_quat(angle):
    """Convert angle to [qw, qz] quaternion representation."""
    return [np.cos(angle/2), np.sin(angle/2)]

# --- 

def run_single_test(lambda_mass, lambda_inertia, render=False, verbose=True):
    """
    Run a single estimation test with specified lambda values.
    lambda_fim = [lambda_mass, 100.0 (CG_x), 100.0 (CG_y), lambda_inertia]
    
    Returns:
        sim: Simulation object
        res: List of [mass_pct_error, cg_x_error_m, cg_y_error_m, inertia_pct_error]
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Testing lambda_mass = {lambda_mass}, lambda_inertia = {lambda_inertia}")
        print(f"{'='*60}")
    
    start = time.time()

    # ------------------------------- SIM SETUP -------------------------------

    # Sim 
    dt = 0.2
    sim_time = 10.0
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
    
    # Controller weights (fixed Q, R; sweep lambda_fim)
    Q = np.diag([1e1, 1e1, 1e2, 1e1, 1e1, 1e0])  # Tracking cost
    R = 1e-1 * np.eye(4)  # Control effort cost (minimal penalty)
    
    # NOTE ON COVARIANCE: Currently using default 0.01*eye(7) in estimate() 
    # This assumes uniform measurement noise across all state dimensions.
    # Could tune this based on actual sensor noise characteristics:
    #   - Position sensors typically more accurate than velocities
    #   - Angular measurements might have different noise than linear
    # For now, uniform weighting is reasonable starting point. Sweep later if needed.
    
    lambda_fim = np.array([[lambda_mass], [100.0], [100.0], [lambda_inertia]])
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
    sim.run(sim_time, render=False, real_time=False)
    

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
    
    # Ground truth
    composite_mass, composite_cg, composite_inertia = sim.get_composite_mass_properties()
    
    # Run estimation
    tht_hat = agent.estimate(oracle_data=agent_data, update_model=False)
    
    # Frame transformation (CG from world to body frame)
    final_state = state_hist[-1]
    agent_pos_world = final_state[:2]
    qw, qz = final_state[2], final_state[3]
    R_body_to_world = np.array([
        [qw**2 - qz**2, -2*qw*qz],
        [2*qw*qz, qw**2 - qz**2]
    ])
    cg_world_relative = composite_cg - agent_pos_world
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
    print(f"  Mass: {mass_error:.3f} kg ({mass_pct_error:.1f}%)")
    print(f"  CG_x: {cg_x_error_body:.4f} m")
    print(f"  CG_y: {cg_y_error_body:.4f} m")
    if composite_inertia != 0:
        print(f"  Inertia: {inertia_error:.4f} kg*m^2 ({inertia_pct_error:.1f}%)")
    else:
        print(f"  Inertia: {inertia_error:.4f} kg*m^2 (GT is zero!)")
    
    # Plot
    # sim.plot_trajectories()
    # sim.plot_states()
    # sim.plot_controls()

    end = time.time()
    print(f"\nSimulation and estimation took {end - start:.2f} seconds.")

    return sim, res


if __name__ == "__main__":
    
    # Grid of lambda values: 5x5 grid
    # Mass weight: 5 log-spaced points from 0.1 to 1000
    # Inertia weight: 5 log-spaced points from 0.1 to 1000
    # CG weights: frozen at 100
    
    lambda_mass_values = np.logspace(-1, 3, 5)  # [0.1, 1, 10, 100, 1000]
    lambda_inertia_values = np.logspace(-1, 3, 5)  # [0.1, 1, 10, 100, 1000]
    
    print(f"Lambda mass values: {lambda_mass_values}")
    print(f"Lambda inertia values: {lambda_inertia_values}")
    print(f"Total tests: {len(lambda_mass_values) * len(lambda_inertia_values)}")
    
    # Create meshgrid for results
    n_mass = len(lambda_mass_values)
    n_inertia = len(lambda_inertia_values)
    
    mass_err_grid = np.zeros((n_mass, n_inertia))
    cg_x_err_grid = np.zeros((n_mass, n_inertia))
    cg_y_err_grid = np.zeros((n_mass, n_inertia))
    inertia_err_grid = np.zeros((n_mass, n_inertia))
    
    # Run tests
    test_count = 0
    total_tests = n_mass * n_inertia
    
    for i, lambda_mass in enumerate(lambda_mass_values):
        for j, lambda_inertia in enumerate(lambda_inertia_values):
            test_count += 1
            print(f"\n{'='*70}")
            print(f"Test {test_count}/{total_tests}: lambda_mass={lambda_mass:.2f}, lambda_inertia={lambda_inertia:.2f}")
            print(f"{'='*70}")
            
            sim, res = run_single_test(lambda_mass, lambda_inertia, render=False, verbose=True)
            
            mass_err_grid[i, j] = res[0]
            cg_x_err_grid[i, j] = res[1]
            cg_y_err_grid[i, j] = res[2]
            inertia_err_grid[i, j] = res[3]
    
    # Create 2D heatmap plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Mass error heatmap
    im0 = axes[0, 0].contourf(lambda_inertia_values, lambda_mass_values, mass_err_grid, levels=20, cmap='viridis')
    axes[0, 0].set_xscale('log')
    axes[0, 0].set_yscale('log')
    axes[0, 0].set_xlabel('λ_inertia', fontsize=12)
    axes[0, 0].set_ylabel('λ_mass', fontsize=12)
    axes[0, 0].set_title('Mass Error (%) Heatmap', fontsize=13, fontweight='bold')
    plt.colorbar(im0, ax=axes[0, 0])
    
    # Inertia error heatmap
    im1 = axes[0, 1].contourf(lambda_inertia_values, lambda_mass_values, inertia_err_grid, levels=20, cmap='viridis')
    axes[0, 1].set_xscale('log')
    axes[0, 1].set_yscale('log')
    axes[0, 1].set_xlabel('λ_inertia', fontsize=12)
    axes[0, 1].set_ylabel('λ_mass', fontsize=12)
    axes[0, 1].set_title('Inertia Error (%) Heatmap', fontsize=13, fontweight='bold')
    plt.colorbar(im1, ax=axes[0, 1])
    
    # CG_x error heatmap
    im2 = axes[1, 0].contourf(lambda_inertia_values, lambda_mass_values, np.abs(cg_x_err_grid), levels=20, cmap='viridis')
    axes[1, 0].set_xscale('log')
    axes[1, 0].set_yscale('log')
    axes[1, 0].set_xlabel('λ_inertia', fontsize=12)
    axes[1, 0].set_ylabel('λ_mass', fontsize=12)
    axes[1, 0].set_title('|CG_x Error| (m) Heatmap', fontsize=13, fontweight='bold')
    plt.colorbar(im2, ax=axes[1, 0])
    
    # CG_y error heatmap
    im3 = axes[1, 1].contourf(lambda_inertia_values, lambda_mass_values, np.abs(cg_y_err_grid), levels=20, cmap='viridis')
    axes[1, 1].set_xscale('log')
    axes[1, 1].set_yscale('log')
    axes[1, 1].set_xlabel('λ_inertia', fontsize=12)
    axes[1, 1].set_ylabel('λ_mass', fontsize=12)
    axes[1, 1].set_title('|CG_y Error| (m) Heatmap', fontsize=13, fontweight='bold')
    plt.colorbar(im3, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig('non_uniform_lambda_heatmaps.png', dpi=150)
    print("\nHeatmap plot saved: non_uniform_lambda_heatmaps.png")
    plt.show()
    
    # --- ZOOMED HEATMAPS (clipped to 0-30% range for better visualization) ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Mass error (clipped to 0-30%)
    mass_err_clipped = np.clip(mass_err_grid, 0, 30)
    im0 = axes[0, 0].contourf(lambda_inertia_values, lambda_mass_values, mass_err_clipped, levels=20, cmap='viridis')
    axes[0, 0].set_xscale('log')
    axes[0, 0].set_yscale('log')
    axes[0, 0].set_xlabel('λ_inertia', fontsize=12)
    axes[0, 0].set_ylabel('λ_mass', fontsize=12)
    axes[0, 0].set_title('Mass Error (%) - Clipped to 0-30%', fontsize=13, fontweight='bold')
    plt.colorbar(im0, ax=axes[0, 0])
    
    # Inertia error (clipped to 0-30%)
    inertia_err_clipped = np.clip(inertia_err_grid, 0, 30)
    im1 = axes[0, 1].contourf(lambda_inertia_values, lambda_mass_values, inertia_err_clipped, levels=20, cmap='viridis')
    axes[0, 1].set_xscale('log')
    axes[0, 1].set_yscale('log')
    axes[0, 1].set_xlabel('λ_inertia', fontsize=12)
    axes[0, 1].set_ylabel('λ_mass', fontsize=12)
    axes[0, 1].set_title('Inertia Error (%) - Clipped to 0-30%', fontsize=13, fontweight='bold')
    plt.colorbar(im1, ax=axes[0, 1])
    
    # CG_x error
    cg_x_err_clipped = np.clip(np.abs(cg_x_err_grid), 0, 0.2)
    im2 = axes[1, 0].contourf(lambda_inertia_values, lambda_mass_values, cg_x_err_clipped, levels=20, cmap='viridis')
    axes[1, 0].set_xscale('log')
    axes[1, 0].set_yscale('log')
    axes[1, 0].set_xlabel('λ_inertia', fontsize=12)
    axes[1, 0].set_ylabel('λ_mass', fontsize=12)
    axes[1, 0].set_title('|CG_x Error| (m) - Clipped to 0-0.2m', fontsize=13, fontweight='bold')
    plt.colorbar(im2, ax=axes[1, 0])
    
    # CG_y error
    cg_y_err_clipped = np.clip(np.abs(cg_y_err_grid), 0, 0.2)
    im3 = axes[1, 1].contourf(lambda_inertia_values, lambda_mass_values, cg_y_err_clipped, levels=20, cmap='viridis')
    axes[1, 1].set_xscale('log')
    axes[1, 1].set_yscale('log')
    axes[1, 1].set_xlabel('λ_inertia', fontsize=12)
    axes[1, 1].set_ylabel('λ_mass', fontsize=12)
    axes[1, 1].set_title('|CG_y Error| (m) - Clipped to 0-0.2m', fontsize=13, fontweight='bold')
    plt.colorbar(im3, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig('non_uniform_lambda_heatmaps_zoomed.png', dpi=150)
    print("Zoomed heatmap plot saved: non_uniform_lambda_heatmaps_zoomed.png")
    plt.show()

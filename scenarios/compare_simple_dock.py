# scenarios/compare_simple_dock.py

"""
Comparison test: Run parameter estimation with different FIM weight configurations
and different sampling rates to compare estimation accuracy.

Configurations:
1. Baseline: lambda_fim = [1, 1, 1, 1] (equal weights on mass, cg_x, cg_y, inertia), dt=1.0
2. High mass/inertia info: lambda_fim = [10, 1, 1, 10], dt=1.0
3. High CG info: lambda_fim = [1, 10, 10, 1], dt=1.0  
4. Fine sampling: lambda_fim = [1, 1, 1, 1], dt=0.2
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


def run_single_config(config_name, lambda_fim, dt, sim_time=5.0, target_dist=2.5):
    """
    Run a single configuration and return estimation errors.
    
    Args:
        config_name: Name of configuration for printing
        lambda_fim: FIM weight vector [4x1] for [mass, cg_x, cg_y, inertia]
        dt: Agent timestep
        sim_time: Total simulation time
        target_dist: Target distance to travel north
    
    Returns:
        Dictionary with ground truth, estimates, and errors
    """
    print(f"\n{'='*60}")
    print(f"Running: {config_name}")
    print(f"  lambda_fim: {lambda_fim.flatten()}")
    print(f"  dt: {dt} s")
    print(f"  sim_time: {sim_time} s")
    print(f"  target_dist: {target_dist} m")
    print(f"{'='*60}\n")
    
    # --- SIMULATION ---
    sim = DMPC_Sim(dt)
    
    # --- SINGLE BLOCK STRUCTURE ---
    blocks = [Unit_Block("block_0", size=1.0)]
    connections = []
    sim.build_superstructure(blocks, connections)
    sim.compile()
    
    # Get block position
    attach = sim.get_attachment_points('block_0')
    target_z = attach[2]
    print(f"Block position: ({attach[0]:.1f}, {attach[1]:.1f})")
    print(f"Target Z for docking: {target_z}")
    
    # --- AGENT ---
    mass = 18.48
    inertia = 0.1462
    moment_arm = 0.12
    cg = np.array([0, 0])
    model = excitation_model(mass, inertia, moment_arm, cg)
    
    heading = 0  # Face north
    x0 = np.array([attach[0], attach[1], *angle_to_quat(heading), 0, 0, 0])
    x_cmd = np.array([attach[0], attach[1] + target_dist, *angle_to_quat(heading), 0, 0, 0])
    
    # Controller parameters
    Q = np.diag([5e1, 5e1, 1e3, 1e2, 1e2, 1e1])
    R = 1e-1 * np.eye(4)
    ctrl_params = {
        'x_cmd': x_cmd, 
        'H': 10, 
        'Q': Q, 
        'R': R, 
        'max_thrust': 1.5,
        'lambda_fim': lambda_fim
    }
    
    # Build agent with docking (this handles everything properly)
    agent = sim.build_agent("Agent_1", model, x0, 'Excitation_MPC', 
                           target_z=target_z, dock_block="block_0", **ctrl_params)
    
    # --- FINALIZE AND RUN ---
    sim.finalize()
    
    # Get body IDs
    agent_body_id = sim.model.body("Agent_1_floatbot").id
    struct_body_id = sim.model.body("block_0").id
    
    print(f"Structure body mass: {sim.model.body_subtreemass[struct_body_id]:.2f} kg")
    print(f"Agent body mass: {sim.model.body_subtreemass[agent_body_id]:.2f} kg")
    
    print("\nRunning simulation...")
    sim.run(sim_time, render=False)
    
    # --- GET GROUND TRUTH ---
    # Use body_mass and sum structure + all agent bodies
    struct_mass = sim.model.body_mass[struct_body_id]
    
    # Sum all agent body masses
    agent_masses = []
    for i in range(sim.model.nbody):
        body_name = sim.model.body(i).name
        if 'Agent_1' in body_name:
            agent_masses.append(sim.model.body_mass[i])
    agent_mass_total = sum(agent_masses)
    
    gt_mass = struct_mass + agent_mass_total
    
    # Get composite inertia from crb
    struct_crb = sim.data.crb[struct_body_id]
    gt_cg_mass_weighted = struct_crb[6:9]
    gt_cg = gt_cg_mass_weighted[:2] / gt_mass
    gt_inertia = struct_crb[5]  # Izz
    
    # --- PARAMETER ESTIMATION ---
    agent_data = sim.Oracle.data['Agent_1']
    
    # Run estimation
    est_params = agent.estimate(
        oracle_data=agent_data,
        update_model=False
    )
    
    # Calculate errors (absolute difference)
    mass_error = est_params[0] - gt_mass
    mass_error_pct = 100 * abs(mass_error) / gt_mass
    cg_x_error = est_params[1] - gt_cg[0]
    cg_y_error = est_params[2] - gt_cg[1]
    inertia_error = est_params[3] - gt_inertia
    inertia_error_pct = 100 * abs(inertia_error) / gt_inertia if gt_inertia != 0 else float('inf')
    
    results = {
        'config': config_name,
        'lambda_fim': lambda_fim.flatten(),
        'dt': dt,
        'gt': {
            'mass': gt_mass,
            'cg_x': gt_cg[0],
            'cg_y': gt_cg[1],
            'inertia': gt_inertia
        },
        'est': {
            'mass': est_params[0],
            'cg_x': est_params[1],
            'cg_y': est_params[2],
            'inertia': est_params[3]
        },
        'errors': {
            'mass': mass_error,
            'mass_pct': mass_error_pct,
            'cg_x': cg_x_error,
            'cg_y': cg_y_error,
            'inertia': inertia_error,
            'inertia_pct': inertia_error_pct
        }
    }
    
    # Print results
    print(f"\n{config_name} Results:")
    print(f"Ground Truth:")
    print(f"  Mass: {gt_mass:.3f} kg")
    print(f"  CG: ({gt_cg[0]:.4f}, {gt_cg[1]:.4f}) m")
    print(f"  Inertia: {gt_inertia:.4f} kg*m^2")
    
    print(f"\nEstimated:")
    print(f"  Mass: {est_params[0]:.3f} kg")
    print(f"  CG: ({est_params[1]:.4f}, {est_params[2]:.4f}) m")
    print(f"  Inertia: {est_params[3]:.4f} kg*m^2")
    
    print(f"\nErrors:")
    print(f"  Mass: {mass_error:.3f} kg ({mass_error_pct:.1f}%)")
    print(f"  CG_x: {cg_x_error:.4f} m")
    print(f"  CG_y: {cg_y_error:.4f} m")
    if gt_inertia != 0:
        print(f"  Inertia: {inertia_error:.4f} kg*m^2 ({inertia_error_pct:.1f}%)")
    else:
        print(f"  Inertia: {inertia_error:.4f} kg*m^2 (GT is zero!)")
    
    return results


def print_comparison_table(all_results):
    """Print a comparison table of all configurations."""
    print(f"\n\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}\n")
    
    # Header
    print(f"{'Configuration':<25} {'Mass Err %':<12} {'CG_X Err':<12} {'CG_Y Err':<12} {'Inertia Err %':<15}")
    print(f"{'-'*80}")
    
    # Data rows
    for res in all_results:
        print(f"{res['config']:<25} "
              f"{res['errors']['mass_pct']:>10.1f}% "
              f"{res['errors']['cg_x']:>10.4f} m "
              f"{res['errors']['cg_y']:>10.4f} m "
              f"{res['errors']['inertia_pct']:>13.1f}%")
    
    print(f"{'-'*80}\n")
    
    # Find best configuration for each metric
    best_mass = min(all_results, key=lambda x: abs(x['errors']['mass']))
    best_inertia = min(all_results, key=lambda x: abs(x['errors']['inertia']))
    
    print(f"Best Mass Error:    {best_mass['config']} ({best_mass['errors']['mass']:.3f} kg, {best_mass['errors']['mass_pct']:.1f}%)")
    print(f"Best Inertia Error: {best_inertia['config']} ({best_inertia['errors']['inertia']:.4f} kg*m^2, {best_inertia['errors']['inertia_pct']:.1f}%)")
    print()


def run_comparison():
    """Run all comparison configurations."""
    print("="*80)
    print("PARAMETER ESTIMATION COMPARISON TEST")
    print("="*80)
    
    all_results = []
    
    # Configuration 1: Baseline - equal weights
    config1 = run_single_config(
        config_name="Baseline (λ=[1,1,1,1])",
        lambda_fim=np.array([[1.0], [1.0], [1.0], [1.0]]),
        dt=1.0,
        sim_time=5.0,
        target_dist=2.5
    )
    all_results.append(config1)
    
    # Configuration 2: High mass/inertia info
    config2 = run_single_config(
        config_name="High m,I (λ=[10,1,1,10])",
        lambda_fim=np.array([[10.0], [1.0], [1.0], [10.0]]),
        dt=1.0,
        sim_time=5.0,
        target_dist=2.5
    )
    all_results.append(config2)
    
    # Configuration 3: High CG info
    config3 = run_single_config(
        config_name="High CG (λ=[1,10,10,1])",
        lambda_fim=np.array([[1.0], [10.0], [10.0], [1.0]]),
        dt=1.0,
        sim_time=5.0,
        target_dist=2.5
    )
    all_results.append(config3)
    
    # Configuration 4: Fine sampling
    config4 = run_single_config(
        config_name="Fine dt (λ=[1,1,1,1], dt=0.2)",
        lambda_fim=np.array([[1.0], [1.0], [1.0], [1.0]]),
        dt=0.2,
        sim_time=5.0,
        target_dist=2.5
    )
    all_results.append(config4)
    
    # Print comparison table
    print_comparison_table(all_results)
    
    return all_results


if __name__ == "__main__":
    results = run_comparison()

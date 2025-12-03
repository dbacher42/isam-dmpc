# scenarios/test_random_structure.py

"""
Test scenario: visualize randomly generated superstructures.
No agents - just build and display the structure.
"""

import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from DMPC_Sim import DMPC_Sim
from superstructure import Superstructure_2D
from scenarios.build_random_structures import (
    generate_random_structure,
    generate_snake_structure, 
    generate_branching_structure
)


def test_random_structure(structure_type='growth', num_blocks=12, seed=None, sim_time=10.0):
    """
    Build and visualize a random structure.
    
    Parameters
    ----------
    structure_type : str
        'growth', 'snake', or 'branching'
    num_blocks : int
        Number of blocks in structure
    seed : int, optional
        Random seed for reproducibility
    sim_time : float
        How long to run viewer (default: 10s)
    """
    print(f"=== Random Structure Test ===")
    print(f"Type: {structure_type}, Blocks: {num_blocks}, Seed: {seed}\n")
    
    # Generate structure
    if structure_type == 'growth':
        blocks, connections = generate_random_structure(num_blocks=num_blocks, seed=seed)
    elif structure_type == 'snake':
        blocks, connections = generate_snake_structure(length=num_blocks, seed=seed)
    elif structure_type == 'branching':
        blocks, connections = generate_branching_structure(num_blocks=num_blocks, seed=seed)
    else:
        raise ValueError(f"Unknown structure type: {structure_type}")
    
    # Build simulation
    dt = 0.1
    sim = DMPC_Sim(dt)
    
    # Build superstructure
    sim.build_superstructure(blocks, connections)
    
    # Finalize and run
    sim.finalize()
    
    print(f"\nStructure has {len(blocks)} blocks")
    print(f"Model has {sim.model.njnt} joints, {sim.model.nbody} bodies")
    
    # Get structure mass
    root_block = blocks[0].id
    body_id = sim.model.body(root_block).id
    total_mass = sim.model.body_subtreemass[body_id]
    print(f"Total structure mass: {total_mass:.2f} kg")
    
    print(f"\n--- Running viewer for {sim_time}s ---")
    sim.run(sim_time, render=True, real_time=True)
    
    print("\n=== Done ===")
    return sim


if __name__ == "__main__":

    test_random_structure(structure_type='growth', num_blocks=100, seed=42, sim_time=20.0)
    test_random_structure(structure_type='snake', num_blocks=100, seed=42, sim_time=20.0)
    test_random_structure(structure_type='branching', num_blocks=100, seed=42, sim_time=20.0)
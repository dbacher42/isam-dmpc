# scenarios/test_random_structure.py

"""
Test scenario with randomly generated superstructures.

Grows a structure from a seed block by randomly adding adjacent blocks,
similar to how crystals grow or how a random walk fills space.
"""

import numpy as np
import random
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from superstructure import Unit_Block, Connection, Superstructure_2D


def generate_random_structure(num_blocks=10, seed=None):
    """
    Generate a random connected structure by growing outward from origin.
    
    Uses a "growth" algorithm:
    1. Start with a seed block at (0, 0)
    2. Maintain a frontier of valid adjacent positions
    3. Randomly select from frontier to add next block
    4. Repeat until desired number of blocks reached
    
    Parameters
    ----------
    num_blocks : int
        Number of blocks in the structure (default: 10)
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    blocks : list[Unit_Block]
        List of blocks in the structure
    connections : list[Connection]
        List of connections between adjacent blocks
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Track occupied positions and their block IDs
    occupied = {}  # (x, y) -> block_id
    blocks = []
    connections = []
    
    # Direction vectors and their names
    directions = {
        'north': (0, 1),
        'south': (0, -1),
        'east': (1, 0),
        'west': (-1, 0)
    }
    opposite = {
        'north': 'south',
        'south': 'north',
        'east': 'west',
        'west': 'east'
    }
    
    def get_neighbors(pos):
        """Get all 4 neighboring positions."""
        x, y = pos
        return {
            'north': (x, y + 1),
            'south': (x, y - 1),
            'east': (x + 1, y),
            'west': (x - 1, y)
        }
    
    def get_frontier():
        """Get all unoccupied positions adjacent to the structure."""
        frontier = set()
        for pos in occupied:
            for direction, neighbor in get_neighbors(pos).items():
                if neighbor not in occupied:
                    frontier.add(neighbor)
        return list(frontier)
    
    def get_connection_to_structure(new_pos):
        """Find which existing block the new position connects to."""
        neighbors = get_neighbors(new_pos)
        for direction, neighbor_pos in neighbors.items():
            if neighbor_pos in occupied:
                # new_pos connects via 'direction' to neighbor_pos
                # neighbor_pos connects via opposite direction to new_pos
                return neighbor_pos, opposite[direction], direction
        return None
    
    # Start with seed block at origin
    seed_pos = (0, 0)
    seed_id = "block_0_0"
    blocks.append(Unit_Block(seed_id, size=1.0))
    occupied[seed_pos] = seed_id
    
    # Grow structure
    block_count = 1
    while block_count < num_blocks:
        # Get available positions
        frontier = get_frontier()
        
        if not frontier:
            print(f"Warning: No more valid positions. Stopping at {block_count} blocks.")
            break
        
        # Randomly select next position
        new_pos = random.choice(frontier)
        x, y = new_pos
        
        # Create block
        block_id = f"block_{x}_{y}"
        blocks.append(Unit_Block(block_id, size=1.0))
        occupied[new_pos] = block_id
        
        # Find and create connection
        conn_info = get_connection_to_structure(new_pos)
        if conn_info:
            neighbor_pos, neighbor_face, new_face = conn_info
            neighbor_id = occupied[neighbor_pos]
            connections.append(Connection(neighbor_id, neighbor_face, block_id, new_face))
        
        block_count += 1
    
    print(f"Generated structure with {len(blocks)} blocks and {len(connections)} connections")
    
    return blocks, connections


def generate_snake_structure(length=10, seed=None):
    """
    Generate a snake-like structure that winds randomly.
    
    Parameters
    ----------
    length : int
        Number of blocks in the snake
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    blocks, connections
    """
    if seed is not None:
        random.seed(seed)
    
    directions = {
        'north': (0, 1),
        'south': (0, -1),
        'east': (1, 0),
        'west': (-1, 0)
    }
    opposite = {
        'north': 'south',
        'south': 'north',
        'east': 'west',
        'west': 'east'
    }
    
    occupied = set()
    blocks = []
    connections = []
    
    # Start at origin
    pos = (0, 0)
    block_id = "block_0_0"
    blocks.append(Unit_Block(block_id, size=1.0))
    occupied.add(pos)
    prev_id = block_id
    
    for i in range(1, length):
        # Get valid directions (not already occupied)
        valid_dirs = []
        for direction, delta in directions.items():
            new_pos = (pos[0] + delta[0], pos[1] + delta[1])
            if new_pos not in occupied:
                valid_dirs.append((direction, new_pos))
        
        if not valid_dirs:
            print(f"Snake trapped at block {i}. Stopping.")
            break
        
        # Choose random direction
        direction, new_pos = random.choice(valid_dirs)
        x, y = new_pos
        
        # Create block and connection
        block_id = f"block_{x}_{y}"
        blocks.append(Unit_Block(block_id, size=1.0))
        connections.append(Connection(prev_id, direction, block_id, opposite[direction]))
        
        occupied.add(new_pos)
        pos = new_pos
        prev_id = block_id
    
    print(f"Generated snake with {len(blocks)} blocks")
    return blocks, connections


def generate_branching_structure(num_blocks=15, branch_prob=0.3, seed=None):
    """
    Generate a tree-like branching structure.
    
    Grows like a tree with random branching. Each step either continues
    the current branch or starts a new branch from a random existing block.
    
    Parameters
    ----------
    num_blocks : int
        Target number of blocks
    branch_prob : float
        Probability of starting a new branch vs continuing current one
    seed : int, optional
        Random seed
        
    Returns
    -------
    blocks, connections
    """
    if seed is not None:
        random.seed(seed)
    
    directions = {
        'north': (0, 1),
        'south': (0, -1),
        'east': (1, 0),
        'west': (-1, 0)
    }
    opposite = {
        'north': 'south',
        'south': 'north',
        'east': 'west',
        'west': 'east'
    }
    
    occupied = {}  # pos -> block_id
    blocks = []
    connections = []
    
    # Start at origin
    pos = (0, 0)
    block_id = "block_0_0"
    blocks.append(Unit_Block(block_id, size=1.0))
    occupied[pos] = block_id
    
    # Track tips of branches for growth
    branch_tips = [pos]
    
    while len(blocks) < num_blocks:
        # Decide: continue current tip or branch from random block
        if random.random() < branch_prob and len(occupied) > 1:
            # Start new branch from random existing block
            branch_pos = random.choice(list(occupied.keys()))
        else:
            # Continue from a tip
            if not branch_tips:
                branch_tips = list(occupied.keys())
            branch_pos = random.choice(branch_tips)
        
        # Find valid growth directions from chosen position
        valid_dirs = []
        for direction, delta in directions.items():
            new_pos = (branch_pos[0] + delta[0], branch_pos[1] + delta[1])
            if new_pos not in occupied:
                valid_dirs.append((direction, new_pos))
        
        if not valid_dirs:
            # This position is blocked, remove from tips
            if branch_pos in branch_tips:
                branch_tips.remove(branch_pos)
            continue
        
        # Grow in random direction
        direction, new_pos = random.choice(valid_dirs)
        x, y = new_pos
        
        # Create block and connection
        new_block_id = f"block_{x}_{y}"
        blocks.append(Unit_Block(new_block_id, size=1.0))
        connections.append(Connection(occupied[branch_pos], direction, new_block_id, opposite[direction]))
        
        occupied[new_pos] = new_block_id
        
        # Update branch tips
        if branch_pos in branch_tips:
            branch_tips.remove(branch_pos)
        branch_tips.append(new_pos)
    
    print(f"Generated branching structure with {len(blocks)} blocks")
    return blocks, connections


# --- TEST / DEMO ---

if __name__ == "__main__":
    print("=== Random Structure Generation Demo ===\n")
    
    # Test growth algorithm
    print("1. Growth algorithm (num_blocks=12, seed=42):")
    blocks, connections = generate_random_structure(num_blocks=12, seed=42)
    print(f"   Blocks: {[b.id for b in blocks]}")
    print(f"   Connections: {[(c.id_a, c.face_a, c.id_b, c.face_b) for c in connections]}\n")
    
    # Test snake algorithm
    print("2. Snake algorithm (length=8, seed=42):")
    blocks, connections = generate_snake_structure(length=8, seed=42)
    print(f"   Blocks: {[b.id for b in blocks]}")
    print(f"   Connections: {[(c.id_a, c.face_a, c.id_b, c.face_b) for c in connections]}\n")
    
    # Test branching algorithm
    print("3. Branching algorithm (num_blocks=10, branch_prob=0.3, seed=42):")
    blocks, connections = generate_branching_structure(num_blocks=10, branch_prob=0.3, seed=42)
    print(f"   Blocks: {[b.id for b in blocks]}")
    print(f"   Connections: {[(c.id_a, c.face_a, c.id_b, c.face_b) for c in connections]}\n")
    
    print("=== Done ===")

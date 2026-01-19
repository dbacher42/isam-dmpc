# build_ring.py

# -------------------------------------------------------------------------------------------------

from superstructure import *

# -------------------------------------------------------------------------------------------------

def build_ring(size=5):
    """ Build an NxN ring of blocks (hollow interior, 1-block thick edge). """
    
    blocks      = []
    connections = []
    positions   = []
    
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
    
    # Create blocks
    for x, y in positions:
        block_id = f"block_{x}_{y}"
        blocks.append(Unit_Block(block_id, size=1.0))
    
    # Create connections (adjacent pairs in sequence)
    for i in range(len(positions)):
        current_pos = positions[i]
        next_pos = positions[(i + 1) % len(positions)]  # Wrap around for closed loop
        
        current_id = f"block_{current_pos[0]}_{current_pos[1]}"
        next_id = f"block_{next_pos[0]}_{next_pos[1]}"
        
        # Determine connection direction
        dx = next_pos[0] - current_pos[0]
        dy = next_pos[1] - current_pos[1]
        
        if dx == 1:      # Moving east
            connections.append(Connection(current_id, "east", next_id, "west"))
        elif dx == -1:   # Moving west  
            connections.append(Connection(current_id, "west", next_id, "east"))
        elif dy == 1:    # Moving north
            connections.append(Connection(current_id, "north", next_id, "south"))
        elif dy == -1:   # Moving south
            connections.append(Connection(current_id, "south", next_id, "north"))
    
    return blocks, connections


def build_centered_ring(size=5, center=(0, 0)):
    """
    Build a ring structure centered at the specified position.
    
    Parameters
    ----------
    size : int
        Ring size (e.g., 7 creates positions from 0 to 6)
    center : tuple
        (x, y) position to center the ring at
        
    Returns
    -------
    blocks, connections
        Ready to use with superstructure, pre-centered
    """
    from superstructure import Superstructure_2D
    
    # Build standard ring
    blocks, connections = build_ring(size)
    
    # Calculate offset needed to center the ring
    # Standard ring goes from (0,0) to (size-1, size-1)
    # So center is at ((size-1)/2, (size-1)/2)
    ring_center_x = (size - 1) / 2
    ring_center_y = (size - 1) / 2
    
    offset_x = center[0] - ring_center_x
    offset_y = center[1] - ring_center_y
    
    # Note: The actual transformation will be applied in the superstructure
    # This function just returns the blocks and offset information
    return blocks, connections, (offset_x, offset_y)

# def test_ring():
#     """Test the ring structure."""
#     size = 7
#     print(f"Creating {size}x{size} ring structure...")
    
#     blocks, connections = build_ring(size)
    
#     print(f"Created {len(blocks)} blocks")
#     print(f"Created {len(connections)} connections")
    
#     # Generate MuJoCo model
#     structure = Superstructure_2D()
#     structure.generate_model(blocks, connections)
#     data = mujoco.MjData(structure.model)
    
#     print("Model created successfully!")
#     print(f"Bodies: {structure.model.nbody}")
#     print(f"Joints: {structure.model.njnt}")
    
#     # Launch viewer
#     viewer.launch(structure.model, data)

# if __name__ == "__main__":
#     test_ring()
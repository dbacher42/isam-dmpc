# superstructure.py 

# -------------------------------------------------------------------------------------------------

import mujoco
from typing import List, Tuple

# -------------------------------------------------------------------------------------------------

class Unit_Block:
    
    """
        Basic building block for the 2D superstructure. 

        Mass properties used for internal calculation within DMPC sim
        to compare against MuJoCo simulation results.
    """

    def __init__(self, 
                 id  : str,
                 size: float = 1.0):
        
        self.id   = id
        self.size = size


# ---

class Connection:

    """
        Define a connection between two block faces.    
    """

    def __init__(self, block_a_id: str, face_a: str, 
                       block_b_id: str, face_b: str):

        self.id_a    = block_a_id
        self.face_a  = face_a
        self.id_b    = block_b_id
        self.face_b  = face_b
        
        # self.joint_type = joint_type                  # weld/slide constraints applied in model generation 


# --- 

class Superstructure_2D:

    """
        Generates the MuJoCo model for the superstructure from specified blocks and connections. 

        Note: all blocks are rigidly connected in this implementation. MuJoCo handles this by default, so 
        we do not need to explicitly define weld joints between blocks. Flexibile connections will require
        joint definitions.
    """

    def __init__(self, name=None):

        self.name = name

        # Formatting
        self.block_rgba = [0.2, 0.7, 0.9, 1.0]  

    # --- 

    def _connect_blocks(self, 
                       block_a_id: str, face_a: str, 
                       block_b_id: str, face_b: str):
        
        """ Create a connection between two blocks. """
        
        return Connection(block_a_id, face_a, block_b_id, face_b)

    # ---

    def generate_model(self,
                       blocks     : List[Unit_Block], 
                       connections: List[Connection]):
        
        """ Generate MuJoCo model from blocks and connections. """

        STRUCTURE = mujoco.MjSpec()

        # Start root block and body 
        root = blocks[0]
        base = STRUCTURE.worldbody.add_body(name=root.id)
        base.add_geom(
                      type=mujoco.mjtGeom.mjGEOM_BOX,
                      size=[root.size/2, root.size/2, root.size/2],
                      rgba=self.block_rgba
                     )
        
        # Enforce planar motion for entire structure 
        base.add_joint(name="x"    , type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[1, 0, 0])
        base.add_joint(name="y"    , type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 1, 0])  
        base.add_joint(name="theta", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 0, 1])

        # Add blocks 
        added = {root.id : base} 
        for c in connections:

            # Select parent and child
            if c.id_a in added and c.id_b not in added:
                parent_id   = c.id_a
                child_id    = c.id_b
                parent_face = c.face_a

            elif c.id_b in added and c.id_a not in added:
                parent_id   = c.id_b
                child_id    = c.id_a
                parent_face = c.face_b
                
            else: continue  

            parent = added[parent_id]
            parent_block = next(b for b in blocks if b.id == parent_id)
            child_block  = next(b for b in blocks if b.id == child_id)

            # Offset to add child block at target face of parent 
            offsets = {"north": (0,  parent_block.size), 
                       "south": (0, -parent_block.size), 
                       "east" : ( parent_block.size, 0), 
                       "west" : (-parent_block.size, 0)} 
            offset = offsets[parent_face]
            
            # Add child body
            child = parent.add_body(
                                    name=child_id,
                                    pos=[offset[0], offset[1], 0]
                                   )
            child.add_geom(
                           type=mujoco.mjtGeom.mjGEOM_BOX,                                 # NO joints, RIGID by default 
                           size=[child_block.size/2, child_block.size/2, child_block.size/2],
                           rgba=self.block_rgba
                          )
            added[child_id] = child
        
        # --- 
        
        self.model = STRUCTURE.compile()

        return 
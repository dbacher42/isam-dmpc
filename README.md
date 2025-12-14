# DMPC_Sim: Distributed Model Predictive Control for Space Structure Assembly

A simulation framework for testing Distributed Model Predictive Control (DMPC) techniques for autonomous, multi-agent, in-orbit assembly missions. The core methodology involves using information-aware excitation trajectories to perform system identifcation tasks on superstructures with unknown mass properties.

## Overview

This simulation has two primary requirements:
1. **Generate and solve for information-aware excitation trajectories** in a realistic physics environment
2. **Enable multiple agents** to use this approach for coordinated transport, stabilization, or system identification missions 

The simulation is built on top of **MuJoCo**, leveraging its physics engine to facilitate the contact dynamics and constraint handling necessary for proper agent-structure docking. Rigid-body welds anchor the agents to the superstructure, with MuJoCo coupling the joints (degrees of freedom) for both bodies to ensure accurate force and torque transfer at the docking sites.

Each floatbot agent follows a planar 2D dynamics model and uses **CasADI** with **IPOPT** to solve a nonlinear optimal control problems (NL OCP) within its MPC controller. The superstructure is dynamically generated from block primitives and connection specifications, allowing for predefined shapes (rings, lines) as well as procedurally generated random structures with branches and gaps.

A simulation executive enforces synchronous update across the entire assembly during runtime, while an Oracle maintains global truth data that can be provided to agents at will. 

### Architecture

**Simulation Executive** ([DMPC_Sim.py](DMPC_Sim.py))
- Oversees superstructure and agent instantiation
- User-defined `dt` sets the interval for each control computation 
- MuJoCo engine steps forward only once all agents have registered their control actions
- Physics evolve at default MuJoCo `dt=0.002s`
- Agents extract state information from MuJoCo / Oracle truth (no estimation or sensing for now)

**Oracle** ([Oracle.py](Oracle.py))
- Maintains continuous ground truth data for the entire simulation
- Provides state history for offline parameter estimation
- Acts as a multi-agent manager to facilitate information sharing between agents 

**Agents** ([agents/](agents/))
- Each agent can have a unique dynamics model and controller 
- Supports multiple controller types:
  - `Excitation_MPC`: Information-aware controller using FIM for parameter estimation
  - `Basic_MPC`: Standard MPC without information objectives
  - `LQR`: Linear Quadratic Regulator for simple control tasks
- Agents can update mass property estimates online during estimation missions if desired 

**Superstructure** ([superstructure/](superstructure/))
- Built from `Unit_Block` primitives with configurable size and mass
- Blocks connected via `Connection` objects specifying attachment faces (north/south/east/west)
- MuJoCo handles rigid connections by default for welded assemblies
- Enforces planar (2D) motion constraints

**Scenarios** ([scenarios/](scenarios/))
- Pre-configured test scenarios for different mission types
- Includes parameter estimation, docking tests, and formation control scenarios
- Includes simple unit block, ring, and procedurally generated structures 

## Key Features

### Information-Aware Trajectory Generation
The framework implements a Fisher Information Matrix (FIM) approach to generate optimal excitation trajectories for parameter estimation. The FIM is considered in an augmented cost function that uses these information parameters to maximize the observability of (minimize the unknown in) target mass properties:

- **Parameters of Interest**: Mass, CG offset (x,y), moment of inertia
- **λ_FIM weights**: User-configurable weights for each parameter to prioritize specific estimation objectives
- **Online Adaptation**: Agents can recompute trajectories using updated mass property estimates

### Docking Mechanism
Agents dock to superstructure blocks using MuJoCo's weld constraints:
- Agents operate in the 2D plane directly underneath the assembly, and dock to the bottom face of a block
- Automatic z-offset computation to position agents underneath blocks
- Rigid coupling of agent and structure joints at docking site
- Proper force/torque transfer through the assembly

### Parameter Estimation Pipeline
The standard workflow follows an **(offline plan) → (act) → (estimate)** procedure:

1. **Offline Planning**: Agent computes optimal excitation trajectory when first added to structure
   - Uses currently assumed mass property values
   - Dynamics propagation in cost function uses these estimates
   
2. **Execution**: Trajectory executed in MuJoCo with true compound mass properties
   - Agent-superstructure assembly evolves with actual physics
   - Oracle records full state history
   
3. **Estimation**: Agent uses state history to build regression matrices
   - Produces updated mass property estimates
   - Can be used for subsequent trajectory recomputation (online adaptation)

Note: Current implementation focuses on initial trajectory computation and execution. Online parameter updating during control execution is allowed but not yet fully evaluated.

## Installation

### Requirements
- Python 3.8+
- MuJoCo 2.0+
- CasADI
- NumPy
- Matplotlib (for visualization)

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd DMPC_Sim

# Install dependencies
pip install mujoco casadi numpy matplotlib

# Verify installation
python -c "import mujoco; import casadi; print('Dependencies installed successfully')"
```

## Usage

### Basic Simulation Example

```python
from DMPC_Sim import DMPC_Sim
from superstructure import Unit_Block, Connection
from agents import Agent, excitation_model, Excitation_MPC
import numpy as np

# ------------------------------------------------------------------------
# Initialize simulation

dt = 0.2  # Control timestep
sim = DMPC_Sim(dt)


# ------------------------------------------------------------------------
# Superstructure 

# simple block 
blocks = [Unit_Block("block_0", size=1.0)]
connections = []
sim.build_superstructure(blocks, connections)
sim.compile()                                   # compile here to add structure spec to full env 

# get docking site 
attach = sim.get_attachment_points('block_0')
target_z = attach[2]                            # target is lower face of block 


# ------------------------------------------------------------------------
# Agent

# system model 
mass = 18.48
inertia = 0.1462
moment_arm = 0.12
model = excitation_model(mass, inertia, moment_arm, cg=np.array([0, 0]))

# target state   
x0    = np.array([0, -2, 1, 0, 0, 0, 0])  
x_cmd = np.array([0, 0, 1, 0, 0, 0])  

# controller 
H = 10 
Q = np.diag([1e1, 1e1, 1e2, 1e1, 1e1, 1e0])
R = 1e-1 * np.eye(4)
lambda_fim=np.ones((4,1))
covariance  = 0.01*np.eye(7)
ctrl_params = {'x_cmd' : x_cmd, 'H' : 10, 'Q' : Q, 'R' : R, 'lambda_fim' : lambda_fim, 'covariance' : covariance}

# have the sim add the agent to handle docking to superstructure 
sim.build_agent("Agent_1", model, x0, 'Excitation_MPC', target_z=target_z, dock_block="block_0", **ctrl_params)

# ------------------------------------------------------------------------
# Run simulation

sim.finalize()
sim.run(time=10.0, render=True, real_time=True)
```

### Structure Types

**Unit Block** - Single 1m³ block (default: 5kg)
```python
blocks = [Unit_Block("block_0", size=1.0)]
connections = []
```

**Ring Formation** - Hollow square structure
```python
from worlds import build_ring
blocks, connections = build_ring(size=5)  # 5x5 ring (16 blocks)
```

**Random Structure** - Procedurally generated with branches
```python
from build_random_structures import generate_random_structure
blocks, connections = generate_random_structure(num_blocks=50, seed=42)
```

## Project Submission: Parameter Estimation Tests

For the final project submission, the following parameter estimation scenarios are tested to demonstrate the framework's capabilities:

### 1. Simple Docking Test ([estimate_simple_dock.py](scenarios/estimate_simple_dock.py))
**Purpose**: Evaluate parameter estimation on a single-block structure with λ_FIM sensitivity analysis

**Structure**: 1 block (10kg mass)  
**Agent Configuration**: 1 agent docked to bottom face  
**Test**: Sweeps through different λ_FIM values to assess information weighting impact

**Run**:
```bash
python scenarios/estimate_simple_dock.py
```

### 2. Ring Structure Test ([estimate_simple_ring.py](scenarios/estimate_simple_ring.py))
**Purpose**: Validate estimation on multi-block symmetric structure

**Structure**: 5×5 hollow ring (16 blocks, 80kg total mass)  
**Agent Configuration**: 1 agent docked to ring perimeter  
**Test**: Parameter estimation with increased structure complexity

**Run**:
```bash
python scenarios/estimate_simple_ring.py
```

### 3. Random Structure Test ([estimate_random_structure.py](scenarios/estimate_random_structure.py))
**Purpose**: Test estimation on irregular, asymmetric structures

**Structure**: 50 blocks with random branching pattern (250kg total mass)  
**Agent Configuration**: 1 agent docked to random block  
**Test**: Parameter estimation on complex, irregular geometry

**Run**:
```bash
python scenarios/estimate_random_structure.py
```

### Interpreting Results

All three scenarios output:
- **Percentage errors** for each estimated parameter (mass, CG_x, CG_y, inertia)
- **Ground truth vs. estimated values** with absolute differences
- **State trajectory plots** (position, velocity, orientation)

The tests validate that the FIM-based excitation approach can succesfully estimate mass properties across structure types, with estimation accuracy highly sensitive to λ_FIM tuning and trajectory richness.

In addition, a non-uniform set of λ_FIM sweeps were also conducted on the simple block test case, in which the mass and inertia information weights are varied against each other over a grid, while the CG weights remain constant. 

## Project Structure

```
DMPC_Sim/
├── DMPC_Sim.py              # Main simulation executive
├── Oracle.py                # Ground truth data manager
├── agents/                  
│   ├── Agent.py             # Floatbot agent class
│   ├── mujoco_agent.xml     # Floatbot MuJoCo definition 
│   ├── controllers/         
│   │   ├── excitation_mpc.py    # FIM-based MPC
│   │   ├── basic_mpc.py         # Standard MPC
│   │   └── lqr.py               # Linear quadratic regulator
│   └── models/              
│       ├── excitation_model.py  # Model with FIM parameters
│       └── basic_model.py       # Standard dynamics
├── superstructure/          
│   └── superstructure.py    # Block-based structure builder
├── scenarios/               
│   ├── estimate_simple_dock.py       # Single block estimation
│   ├── estimate_simple_ring.py       # Ring structure estimation
│   ├── estimate_random_structure.py  # Random structure estimation
│   └── ...                  
├── worlds/                  # Structure generators
│   └── build_ring.py        
└── viz/                     # Visualization tools
    └── Cartographer.py      
```

## References

- MuJoCo: A physics engine for model-based control (Todorov et al., 2012)
- CasADI: A software framework for nonlinear optimization (Andersson et al., 2019)
- Fisher Information Matrix approaches for parameter estimation in robotics

## License

See [LICENSE](LICENSE) file for details.

## Contact

soval@usc.edu, dbacher@usc.edu

Space Engineering Research Center (SERC)  
University of Southern California

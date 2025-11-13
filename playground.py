import numpy as np
import matplotlib.pyplot as plt

from DMPC_Sim import DMPC_Sim
from model    import FloatbotModel

# -------------------------------------------------------------------------------------------------
#                                           PLAYGROUND
# -------------------------------------------------------------------------------------------------

def scenario_setup():
    """
    Super simple 2-agent test:
    - 10x10 world 
    - Agent 1 (LQR): Bottom left, faces NE, goes to top left, ends facing up
    - Agent 2 (MPC): Bottom right, faces NW, goes to top right, ends facing up
    """
    
    # --- --- --- --- --- PRELIMS --- --- --- --- --- 

    # Simulation parameters
    dt  = 0.1
    sim = DMPC_Sim(dt)
    
    # Cardinal directions
    N , S , E , W  = np.pi/2, 3*np.pi/2,         0,     np.pi
    NE, NW, SW, SE = np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4

    def angle_to_quat(angle):
        """Convert angle to [qw, qz]"""
        return [np.cos(angle/2), np.sin(angle/2)]


    # --- --- --- --- --- AGENT SETUP --- --- --- --- --- 

    # Floatbot model (common)
    mass       = 53.76
    inertia    = 33.11
    moment_arm = 0.12
    cg         = np.array([0, 0])
    model      = FloatbotModel(mass, inertia, moment_arm, cg)
    
    # Controller weights (currently common)
    H = 20
    Q = np.diag([5e1, 5e1, 8e1, 1e1, 1e1, 1e1])
    R = 1e-1 * np.eye(4)
    
    x0s   = [ 
             np.array([-10, -10, *angle_to_quat(E), 0, 0, 0]), # mpc
             np.array([-10,  10, *angle_to_quat(E), 0, 0, 0]),
             np.array([ 10, -10, *angle_to_quat(W), 0, 0, 0]),
             np.array([ 10,  10, *angle_to_quat(W), 0, 0, 0]),

             np.array([-10,   5, *angle_to_quat(E), 0, 0, 0]), # lqr
             np.array([ -5, -10, *angle_to_quat(N), 0, 0, 0]),
             np.array([ 10,  -5, *angle_to_quat(W), 0, 0, 0]),
             np.array([  5,  10, *angle_to_quat(S), 0, 0, 0]),
            ]
    
    xcmds = [ 
             np.array([ 10,   0, *angle_to_quat(N), 0, 0, 0]),
             np.array([  0, -10, *angle_to_quat(N), 0, 0, 0]),
             np.array([  0,  10, *angle_to_quat(S), 0, 0, 0]),
             np.array([-10,   0, *angle_to_quat(S), 0, 0, 0]),

             np.array([  5, -10, *angle_to_quat(S), 0, 0, 0]),
             np.array([ 10,   5, *angle_to_quat(E), 0, 0, 0]),
             np.array([ -5,  10, *angle_to_quat(N), 0, 0, 0]),
             np.array([-10,  -5, *angle_to_quat(S), 0, 0, 0]),
            ]
    
    controllers = [1, 1, 1, 1, 0, 0, 0, 0]  # 1: MPC, 0: LQR


    # --- --- --- --- --- BUILD AGENTS --- --- --- --- --- 

    for i in range(len(x0s)):
        
        x0      = x0s[i]
        x_cmd   = xcmds[i]
        ctrl_id = controllers[i]
        
        if ctrl_id == 0:  
            ctrl_type   = 'LQR'
            ctrl_params = {'x_cmd': x_cmd, 'Q': Q, 'R': R}
        else:            
            ctrl_type   = 'MPC'
            ctrl_params = {'x_cmd': x_cmd, 'H': H, 'Q': Q, 'R': R}
        
        name  = f"Agent_{i+1}_{ctrl_type}"
        sim.build_agent(name, model, x0, ctrl_type, **ctrl_params)
        print(f" Built {name}. Start: {x0[:4]} → Target: {x_cmd[:4]}")

    sim.finalize()
    return sim

# ---

def run_scenario(sim, T):
    """
        Run the given simulation scenario for time T
    """

    print(f"Running simulation for T={T} seconds with {len(sim.agents)} agents...")
    print("=" * 100)

    try:
        sim.run(T)
        
        print("Simulation completed successfully!")
        print("\nFinal Results:")
        print("-" * 100)
        for _, agent in enumerate(sim.agents):
            final_pos = agent.x_history[-1][:2]
            target_pos = agent.controller.x_cmd[:2]
            error = np.linalg.norm(final_pos - target_pos)
            print(f"{agent.name}: Final [{final_pos[0]:.4f}, {final_pos[1]:.4f}] | Target [{target_pos[0]:.4f}, {target_pos[1]:.4f}] | Error {error:.2f}")

    except Exception as e:
        print(f"Simulation failed: {e}")
        return None


# -------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    
    sim = scenario_setup()
    T   = 10  # seconds
    run_scenario(sim, T)

    # --- 

    sim.plot_trajectories()
    sim.plot_states()
    


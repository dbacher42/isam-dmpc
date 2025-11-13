import numpy as np 

# -------------------------------------------------------------------------------------------------

class Oracle():
    """ 
    
        Oracle sees all.

        Knows the true state of the environment and all agents.
        Info stored as: 

            data = { 
                agent_id : {
                    'state'   : [x_0, x_1, ..., x_k],  # history of states
                    'control' : [u_0, u_1, ..., u_k],  # history of controls
                }
            }
        
        Timestep k means after k steps have been taken, so u[k] 
        Can record all history or just current state. 

    """

    def __init__(self):
        
        self.data   = {}
        self.t_hist = [0]


    # --- --- --- --- --- AGENT MANAGEMENT --- --- --- --- --- 

    def add_agent(self, agent):
        """ 
            Track agent data. 
        """
        self.data[agent.name] = {
            'state'      : [agent.x_current],
            'control'    : [],
            'target'     : agent.controller.x_cmd,
            'controller' : agent.controller.name
        }


    # --- --- --- --- --- STATE ACCESS --- --- --- --- ---

    def get_state(self, agent_id, timestep=None):
        """ 
            Get the state of the specified agent at the specified timestep.
            If timestep is None, get the current state.
        """

        if agent_id not in self.data:
            raise ValueError(f"Agent ID {agent_id} not found in Oracle data.")

        if timestep is None:
            return self.data[agent_id]['state'][-1]
        else:
            return self.data[agent_id]['state'][timestep]

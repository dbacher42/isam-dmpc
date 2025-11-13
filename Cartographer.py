import matplotlib
import matplotlib.cm        as     cm 
import matplotlib.pyplot    as     plt
from   matplotlib.animation import FuncAnimation
matplotlib.use('TkAgg')     # Use stable backend for Windows

# -------------------------------------------------------------------------------------------------

class Cartographer:
    """
    
        Handles all animation and visualization. 
    
    """

    def __init__(self):

        self.kolors = {}

        pass

    # --- 

    def _assign_colors(self, agents):
        """Assign colors to all agents for plotting."""
        cmap = cm.get_cmap('turbo')
        num_agents = len(agents)
        for i, agent in enumerate(agents):
            agent.kolor = cmap(i / max(num_agents - 1, 1))
            self.kolors[agent.name] = agent.kolor
        



    # --- --- --- --- --- CORE --- --- --- --- --- 

    # Animation 
    def animate(self, agents=None): 
        pass 

    # Final Trajectory Plot
    def plot_trajectories(self, agents):
        """
        Plot agent trajectories from initial to final states with targets.
        
        Visual elements:
        - Initial state: Clear circle with solid border
        - Final state: Filled circle  
        - Target state: Clear circle with dotted border
        - Trajectory: Dotted line path
        - Heading vectors: Arrows at initial/final/target positions
        """
        import numpy as np
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for agent in agents:
            color = agent.kolor
            x_hist = np.array(agent.x_history)
            
            # Trajectory path (dotted line)
            ax.plot(x_hist[:, 0], x_hist[:, 1], '--', color=color, alpha=0.7, 
                   linewidth=2, label=f'{agent.name}')
            
            # Initial state (clear circle + heading)
            x0 = x_hist[0]
            radius = agent.model.moment_arm
            ax.add_patch(plt.Circle(x0[:2], radius, 
                                   fill=False, edgecolor=color, linewidth=2))
            heading0 = self._get_heading_vector(x0[2], x0[3], radius)
            self._plot_heading_arrow(ax, x0[:2], heading0, color, radius)
            
            # Final state (filled circle + heading)  
            xf = x_hist[-1]
            ax.add_patch(plt.Circle(xf[:2], radius,
                                   fill=True, facecolor=color, alpha=0.7, edgecolor=color))
            headingf = self._get_heading_vector(xf[2], xf[3], radius)
            self._plot_heading_arrow(ax, xf[:2], headingf, color, radius)
            
            # Target (clear circle with dotted edge + heading)
            target = agent.controller.x_cmd
            ax.add_patch(plt.Circle(target[:2], radius,
                                   fill=False, edgecolor=color, linestyle='--', linewidth=2))
            heading_target = self._get_heading_vector(target[2], target[3], radius)
            self._plot_heading_arrow(ax, target[:2], heading_target, color, radius)
        
        # Plot formatting
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_title('Multi-Agent Trajectories')
        
        # Add legend explanation
        fig.text(0.02, 0.02, 'Legend: ○ Initial, ● Final, ⬚ Target, → Heading', fontsize=9)
        
        plt.tight_layout()
        plt.show()
    
    def _get_heading_vector(self, qw, qz, length):
        """
        Get heading vector from quaternion components.
        Consistent with sim.py rotation matrix logic.
        """
        import numpy as np
        
        # Same rotation matrix as sim.py
        R_BI = np.array([
            [qw**2-qz**2, -2*qw*qz],
            [2*qw*qz, qw**2-qz**2]
        ])
        
        # Base vector pointing in +x direction (same as sim.py r_line)
        base_vector = np.array([[length], [0]])
        
        # Apply rotation to get heading vector
        heading_vector = R_BI @ base_vector
        return heading_vector.flatten()  # Return as [dx, dy]
    
    def _plot_heading_arrow(self, ax, center, heading_vector, color, radius):
        """
        Plot heading arrow from circle edge outward.
        """
        import numpy as np
        
        cx, cy = center
        dx, dy = heading_vector
        
        # Normalize heading vector to get direction
        length = np.sqrt(dx**2 + dy**2)
        if length > 0:
            unit_dx, unit_dy = dx/length, dy/length
        else:
            unit_dx, unit_dy = 1.0, 0.0  # Default to +x direction
        
        # Calculate starting point at circle edge
        edge_x = cx + radius * unit_dx
        edge_y = cy + radius * unit_dy
        
        # Scale arrow vector (make it visible but not too long)
        arrow_scale = 0.8  # Adjust this to change arrow length
        arrow_dx = dx * arrow_scale
        arrow_dy = dy * arrow_scale
        
        # Draw arrow from edge outward
        ax.arrow(edge_x, edge_y, arrow_dx, arrow_dy, 
                head_width=0.05, head_length=0.03, 
                fc=color, ec=color, linewidth=2, alpha=0.8)

    # States
    def plot_states(self, agents=None): 
        pass

    # Controls
    def plot_controls(self, agents=None): 
        pass


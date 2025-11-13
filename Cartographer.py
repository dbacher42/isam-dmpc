import matplotlib
import matplotlib.cm        as     cm 
import matplotlib.pyplot    as     plt
from   matplotlib.animation import FuncAnimation
matplotlib.use('TkAgg')     # Use stable backend for Windows
import numpy as np

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

    # --- 

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

    # --- 

    # States
    def plot_states(self, agents=None, layered=True): 
        """
        Plot state trajectories for agents.
        
        Args:
            agents: List of agents to plot. If None, plots all available agents.
            layered: If True, plot all agents on same figure (multi-agent view).
                    If False, create separate figure for each agent.
        """
            
        # Ensure colors are assigned
        self._assign_colors(agents)
        
        # State configuration
        state_info = [
            ('X Position [m]', 0),
            ('Y Position [m]', 1), 
            ('Orientation [rad]', 'angle'),
            ('X Velocity [m/s]', 4),
            ('Y Velocity [m/s]', 5),
            ('Angular Velocity [rad/s]', 6)
        ]
        
        agents_to_plot = [agents] if layered else [[agent] for agent in agents]
        
        for agent_group in agents_to_plot:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # Set title based on mode
            if layered:
                fig.suptitle('Multi-Agent State Trajectories', fontsize=16, fontweight='bold')
            else:
                fig.suptitle(f'{agent_group[0].name} State Trajectory', fontsize=16, fontweight='bold')
            
            # Plot each agent in this group
            for agent in agent_group:
                self._plot_agent_states(agent, axes, state_info, layered)
            
            # Add legends
            self._add_state_legends(axes, layered)
            
            plt.tight_layout()
            plt.show()
    
    def _plot_agent_states(self, agent, axes, state_info, layered):
        """Plot single agent's states on provided axes."""
        
        color = self.kolors[agent.name]
        
        if not (hasattr(agent, 'x_history') and len(agent.x_history) > 0):
            return
            
        # Prepare data
        n_steps = len(agent.x_history)
        dt = 0.1  # TODO: Get from simulation
        time = np.arange(n_steps) * dt
        x_hist = np.array(agent.x_history)
        x_cmd = agent.controller.x_cmd
        
        # Plot each state variable
        for i, (ylabel, state_idx) in enumerate(state_info):
            row, col = divmod(i, 3)
            ax = axes[row, col]
            
            # Extract state data
            state_data, cmd_data = self._extract_state_data(x_hist, x_cmd, state_idx)
            
            # Plot trajectory (solid line)
            label = agent.name if (layered and i == 0) else ("Trajectory" if not layered and i == 0 else "")
            ax.plot(time, state_data, color=color, linewidth=2, label=label)
            
            # Plot commanded state (dashed line)
            target_label = "Target" if (not layered and i == 0) else ""
            ax.axhline(y=cmd_data, color=color, linestyle='--', 
                      alpha=0.7, linewidth=1, label=target_label)
            
            # Formatting (only set once per subplot)
            if not ax.get_ylabel():  # Only set if not already set
                ax.set_ylabel(ylabel)
                ax.grid(True, alpha=0.3)
                if not layered:
                    ax.set_title(f'{ylabel}')
                if row == 1:  # Bottom row
                    ax.set_xlabel('Time [s]')
    
    def _extract_state_data(self, x_hist, x_cmd, state_idx):
        """Extract state and command data for given state index."""
        
        if state_idx == 'angle':
            # Convert quaternion to angle
            qw_hist, qz_hist = x_hist[:, 2], x_hist[:, 3]
            angle_hist = 2 * np.arctan2(qz_hist, qw_hist)
            
            qw_cmd, qz_cmd = x_cmd[2], x_cmd[3]
            angle_cmd = 2 * np.arctan2(qz_cmd, qw_cmd)
            
            return angle_hist, angle_cmd
        else:
            return x_hist[:, state_idx], x_cmd[state_idx]
    
    def _add_state_legends(self, axes, layered):
        """Add appropriate legends based on plotting mode."""
        
        if layered:
            # Agent legend on first subplot
            axes[0,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Style legend on top-right subplot
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='black', linewidth=2, label='Trajectory'),
                Line2D([0], [0], color='black', linewidth=1, linestyle='--', label='Target')
            ]
            axes[0,2].legend(handles=legend_elements, loc='upper right', 
                            frameon=True, fancybox=True, shadow=True)
        else:
            # Simple legend for single agent (first subplot only)
            axes[0,0].legend()

    # ---

    # Controls
    def plot_controls(self, agents=None): 
        pass



    # --- --- --- --- --- UTILS --- --- --- --- --- 
    
    def _get_heading_vector(self, qw, qz, length):
        """
        Get heading vector from quaternion components.
        Consistent with sim.py rotation matrix logic.
        """
        
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
    
    # --- 

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



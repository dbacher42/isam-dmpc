import numpy as np
import matplotlib
import matplotlib.cm        as     cm 
import matplotlib.pyplot    as     plt
from   matplotlib.lines     import Line2D
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
        """
        Animate multi-agent simulation showing all agents moving simultaneously.
        Uses pre-computed trajectories from agent histories.
        """
        
        if agents is None or len(agents) == 0:
            print("No agents provided for animation")
            return
            
        # Ensure colors are assigned
        self._assign_colors(agents)
        
        # Validate that agents have history data
        valid_agents = []
        for agent in agents:
            if hasattr(agent, 'x_history') and len(agent.x_history) > 0:
                valid_agents.append(agent)
            else:
                print(f"Warning: {agent.name} has no trajectory data")
                
        if not valid_agents:
            print("No agents with valid trajectory data found")
            return
            
        print(f"Animating {len(valid_agents)} agents...")
        self._create_multi_agent_animation(valid_agents)

    # ---
    
    def _create_multi_agent_animation(self, agents):
        """Create and display multi-agent animation."""
        
        # Animation parameters
        dt = 0.1  # TODO: Get from simulation
        
        # Determine world bounds from all agent trajectories
        all_positions = []
        max_frames = 0
        
        for agent in agents:
            x_hist = np.array(agent.x_history)
            all_positions.extend(x_hist[:, :2])  # x, y positions
            max_frames = max(max_frames, len(x_hist))
            
        if not all_positions:
            print("No position data found")
            return
            
        positions = np.array(all_positions)
        margin = 2.0
        x_min, x_max = positions[:, 0].min() - margin, positions[:, 0].max() + margin
        y_min, y_max = positions[:, 1].min() - margin, positions[:, 1].max() + margin
        
        # Create figure and setup
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.set_xlabel('X Position [m]')
        ax.set_ylabel('Y Position [m]')
        ax.set_title('Multi-Agent DMPC Animation')
        ax.grid(True, alpha=0.3)
        
        # Create plot elements for each agent
        agent_elements = {}
        r_bot = 0.5  # Floatbot radius for visualization
        
        for agent in agents:
            color = self.kolors[agent.name]
            
            # Create circle for agent body
            agent_patch = plt.Circle((0, 0), r_bot, fc=color, ec='black', alpha=0.7)
            ax.add_patch(agent_patch)
            
            # Create line for heading direction
            dir_line, = ax.plot([], [], color='black', linewidth=3, alpha=0.8)
            
            # Create trajectory trace (optional - shows path taken)
            trace_line, = ax.plot([], [], color=color, linewidth=1, alpha=0.3, linestyle='--')
            
            # Store elements
            agent_elements[agent.name] = {
                'patch': agent_patch,
                'dir_line': dir_line,
                'trace_line': trace_line,
                'history': np.array(agent.x_history)
            }
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=self.kolors[agent.name], 
                                 markersize=10, label=agent.name) for agent in agents]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Time display
        time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=12, 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        def animate_frame(frame_idx):
            """Update function for each animation frame."""
            
            updated_elements = []
            
            # Update each agent
            for agent in agents:
                elements = agent_elements[agent.name]
                history = elements['history']
                
                # Handle agents with different trajectory lengths
                if frame_idx < len(history):
                    # Current state
                    x, y, qw, qz = history[frame_idx, :4]
                    
                    # Update agent position
                    elements['patch'].set_center((x, y))
                    updated_elements.append(elements['patch'])
                    
                    # Update heading direction using same logic as sim.py
                    heading_vec = self._get_heading_vector(qw, qz, r_bot * 1.5)
                    dir_x, dir_y = heading_vec
                    elements['dir_line'].set_data([x, x + dir_x], [y, y + dir_y])
                    updated_elements.append(elements['dir_line'])
                    
                    # Update trajectory trace
                    if frame_idx > 0:
                        trace_x = history[:frame_idx+1, 0]
                        trace_y = history[:frame_idx+1, 1]
                        elements['trace_line'].set_data(trace_x, trace_y)
                    updated_elements.append(elements['trace_line'])
                    
            # Update time display
            time_text.set_text(f'Time: {frame_idx * dt:.2f} s')
            updated_elements.append(time_text)
            
            return updated_elements
        
        # Create and run animation
        print(f"Creating animation with {max_frames} frames...")
        self.multi_agent_animation = FuncAnimation(
            fig, animate_frame, frames=max_frames, 
            interval=dt*1000, blit=True, repeat=True
        )
        
        plt.tight_layout()
        plt.show()

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

    # ---
    
    def _plot_agent_states(self, agent, axes, state_info, layered):
        """Plot single agent's states on provided axes."""
        
        color = self.kolors[agent.name]
        
        if not (hasattr(agent, 'x_history') and len(agent.x_history) > 0):
            return
            
        # Prepare data
        n_steps = len(agent.x_history)
        dt = 0.002  # TODO: Get from simulation
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
    

    # ---

    # Controls
    def plot_controls(self, agents=None, layered=True): 
        """
        Plot control input trajectories for agents.
        
        Args:
            agents: List of agents to plot. If None, plots all available agents.
            layered: If True, plot all agents on same figure (multi-agent view).
                    If False, create separate figure for each agent.
        """
        
        if agents is None:
            print("No agents provided to plot_controls")
            return
            
        # Ensure colors are assigned
        self._assign_colors(agents)
        
        # Control configuration (4 thrusters)
        control_info = [
            ('Thruster 1 [N]', 0),
            ('Thruster 2 [N]', 1), 
            ('Thruster 3 [N]', 2),
            ('Thruster 4 [N]', 3)
        ]
        
        agents_to_plot = [agents] if layered else [[agent] for agent in agents]
        
        for agent_group in agents_to_plot:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))  # 2x2 for 4 thrusters
            
            # Set title based on mode
            if layered:
                fig.suptitle('Multi-Agent Control Inputs', fontsize=16, fontweight='bold')
            else:
                fig.suptitle(f'{agent_group[0].name} Control Inputs', fontsize=16, fontweight='bold')
            
            # Plot each agent in this group
            for agent in agent_group:
                self._plot_agent_controls(agent, axes, control_info, layered)
            
            # Add legends
            self._add_control_legends(axes, layered)
            
            plt.tight_layout()
            plt.show()
    
    # ---

    def _plot_agent_controls(self, agent, axes, control_info, layered):
        """Plot single agent's control inputs on provided axes."""
        
        color = self.kolors[agent.name]
        
        if not (hasattr(agent, 'u_history') and len(agent.u_history) > 0):
            print(f"No control history found for {agent.name}")
            return
            
        # Prepare data
        n_steps = len(agent.u_history)
        dt = 0.1  # TODO: Get from simulation
        time = np.arange(n_steps) * dt
        u_hist = np.array(agent.u_history)  # Shape: (n_steps, 4)
        
        # Plot each control input
        for i, (ylabel, control_idx) in enumerate(control_info):
            row, col = divmod(i, 2)  # 2x2 grid
            ax = axes[row, col]
            
            # Extract control data
            control_data = u_hist[:, control_idx]
            
            # Plot control trajectory
            label = agent.name if (layered and i == 0) else ("Control Input" if not layered and i == 0 else "")
            ax.plot(time, control_data, color=color, linewidth=2, label=label)
            
            # Formatting (only set once per subplot)
            if not ax.get_ylabel():  # Only set if not already set
                ax.set_ylabel(ylabel)
                ax.grid(True, alpha=0.3)
                if not layered:
                    ax.set_title(f'{ylabel}')
                if row == 1:  # Bottom row
                    ax.set_xlabel('Time [s]')
                    
            # Add zero reference line
            ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    
    def _add_control_legends(self, axes, layered):
        """Add appropriate legends for control plots based on plotting mode."""
        
        if layered:
            # Agent legend on first subplot
            axes[0,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            # Simple legend for single agent (first subplot only)
            axes[0,0].legend()



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

    # --- 

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
        
    # --- 
    
    def _add_state_legends(self, axes, layered):
        """Add appropriate legends based on plotting mode."""
        
        if layered:
            # Agent legend on first subplot
            axes[0,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Style legend on top-right subplot
            legend_elements = [
                Line2D([0], [0], color='black', linewidth=2, label='Trajectory'),
                Line2D([0], [0], color='black', linewidth=1, linestyle='--', label='Target')
            ]
            axes[0,2].legend(handles=legend_elements, loc='upper right', 
                            frameon=True, fancybox=True, shadow=True)
        else:
            # Simple legend for single agent (first subplot only)
            axes[0,0].legend()


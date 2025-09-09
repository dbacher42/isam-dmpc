import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

class FloatbotSim():
    def __init__(self, model, controller, x0=np.array([0, 0, 1, 0, 0, 0, 0])):
        '''
        Class for simulating the floatbot controller

        Parameters
        ----------
        model : floatbot kinematics and dynamics model
        controller : floatbot controller
        x0 : initial states
            optional, the default is np.array([0, 0, 1, 0, 0, 0, 0]).
        '''
        # Set simulation parameters
        self.model = model
        self.controller = controller
        self.dt = controller.dt
        
        #Initialize data lists
        self.t_history = [0]
        self.x_history = [x0]
        self.u_history = []
        
        self.x_current = x0
        self.u_current = np.zeros(8)
    
    def step(self, i):
        '''
        Step the simulation forward one time step
        '''
        # Extract parameters
        dt = self.dt
        xdot = self.model.xdot
        control_law = self.controller.solve
        
        # Compute control input
        self.u_current = control_law(self.x_current)
        self.u_history.append(self.u_current)
        
        # Propogate state forward
        x_next = self.x_current + dt*xdot(self.x_current, self.u_current)
        self.x_current = np.array(x_next.full()).flatten()
        
        # Normalize quaternion
        qw = self.x_current[2]
        qz = self.x_current[3]
        q_mag = np.sqrt(qw**2+qz**2)
        self.x_current[2] = qw/q_mag
        self.x_current[3] = qz/q_mag
        
        self.x_history.append(self.x_current)
        
        # Propogate time forward
        self.t_history.append((i+1)*dt)
        
    def run(self, Tf):
        '''
        Run simulation
        '''
        # Extract parameters
        dt = self.dt
        n_steps = int(Tf/dt)
        
        # Run simulation
        for i in range(n_steps):
            self.step(i)
            
        self.animate(-2, 2, -2, 2)
        
    def animate(self, lbx, ubx, lby, uby):
        def rot(q):
            '''
            2x2 z axis rotation matrix from quaternion
            '''
            qw = q[0]
            qz = q[1]
            
            return np.array([
                [qw**2-qz**2, -2*qw*qz],
                [2*qw*qz, qw**2-qz**2]
            ])
        
        def frame(i):
            '''
            Create frame of the animation
            '''
            # Extract state: [rx, ry, qw, qz, vx, vy, omg]
            rx = x_history[i,0]
            ry = x_history[i,1]
            qw = x_history[i,2]
            qz = x_history[i,3]
            R_BI = rot([qw,qz])
            
            bot_patch.set_center((rx, ry))
            dir_end = R_BI@r_line
            dir_line.set_data([rx, rx+dir_end[0,0]], [ry, ry+dir_end[1,0]])
            rcg_I = R_BI@r_cg
            cg.set_data([rx+rcg_I[0,0]], [ry+rcg_I[1,0]])
            time_text.set_text(f'Time = {i*dt:.2f} s')
            
            return bot_patch, dir_line, cg, time_text
            
        # Extract parameters
        x_history = np.array(self.x_history)
        r_bot = self.model.moment_arm
        r_line = np.array([[r_bot],[0]])
        r_cg = np.array(self.model.cg).reshape(2,1)
        dt = self.dt
        
        # Create figure and axis for the animation
        fig, ax = plt.subplots(figsize=(8,4))
        ax.set_xlim(lbx, ubx)
        ax.set_ylim(lby, uby)
        ax.set_aspect('equal')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_title('Floatbot Animation')
        
        # Create plot elements: rectangle for floatbot
        bot_patch = plt.Circle((0, 0), r_bot, fc='b', ec='k')
        ax.add_patch(bot_patch)
        dir_line, = ax.plot([], [], 'c-', lw=3)
        cg, = ax.plot([], [], 'r+', ms=10, mew=3)
        
        # Text annotations for time
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12)
        
        # Create animation object
        self.bot_animation = animation.FuncAnimation(fig, frame, frames=len(x_history), interval=dt*1000, blit=True)
        
if __name__ == "__main__":
    from floatbot_model import FloatbotModel
    from floatbot_lqr import FloatbotLQR
    from floatbot_mpc import FloatbotMPC
    
    # Floatbot parameters
    mass = 16.8
    inertia = .1594
    max_thrust = 1.5
    moment_arm = .12
    cg = np.array([.168, 0])
    
    # States
    rx = 0
    ry = 0
    tht = np.pi/2
    qw = np.cos(tht/2)
    qz = np.sin(tht/2)
    vx = 0
    vy = 0
    wz = 0
    x0 = np.array([rx, ry, qw, qz, vx, vy, wz])
    
    # Commanded States
    rx_cmd = 1
    ry_cmd = 0
    tht_cmd = 0
    qw_cmd = np.cos(tht_cmd/2)
    qz_cmd = np.sin(tht_cmd/2)
    vx_cmd = 0
    vy_cmd = 0
    wz_cmd = 0
    x_cmd = np.array([rx_cmd, ry_cmd, qw_cmd, qz_cmd, vx_cmd, vy_cmd, wz_cmd])
    
    # Controller parameters
    dt = .1
    H = 20
    Q = np.diag([5e1,5e1,8e3,1e1,1e1,1e1])
    R = 1e-1*np.eye(4)
    
    # Simulator parameters
    Tf = 10
    floatbot_model = FloatbotModel(mass, inertia, max_thrust, moment_arm, cg)
    floatbot_lqr = FloatbotLQR(floatbot_model, x_cmd, dt, Q=Q, R=R)
    floatbot_mpc = FloatbotMPC(floatbot_model, x_cmd, dt, H, Q, R)
    lqr_sim = FloatbotSim(floatbot_model, floatbot_lqr, x0)
    mpc_sim = FloatbotSim(floatbot_model, floatbot_mpc, x0)
    
    # Run simulation
    lqr_sim.run(Tf)
    mpc_sim.run(Tf)
import numpy as np
import casadi as cs
import matplotlib.pyplot as plt

def mixer(u, R):
    f_B = cs.vertcat(
        u[0]+u[2],
        u[1]+u[3]
    )
    f_I = R@f_B
    tau = r*(-u[0]-u[1]+u[2]+u[3])
    F = cs.vertcat(f_I, tau)

    return F

def rot(qw, qz):
    R = cs.vertcat(
        cs.horzcat(qw**2-qz**2, -2*qw*qz),
        cs.horzcat(2*qw*qz, qw**2-qz**2)
    )

    return R

def x_dot(x, u, tht):
    '''
    Computes the floatbot state derivatives

    Parameters
    ----------
    x : 7x1 state vector
        x[0] : x position in the inertial frame (m)
        x[1] : y position in the inertial frame (m)
        x[2] : qw real quaternion component
        x[3] : qz imaginiary quaternion component 
        (note: normalize quaternion to prevent numerical errors)
        x[4] : x velocity in the body frame (m/s)
        x[5] : y velocity in the vody frame (m/s)
        x[6] : z axis rotational velocity (rad/s)
    u : 4x1 control vector
        u[i] : thrust of thruster i (N)
    tht : 4x1 parameter vector
        tht[0] : mass (kg)
        tht[1] : x cg offset in the body frame (m)
        tht[2] : y cg offset in the body frame (m)
        tht[3] : z axis moment of inertia (kg*m**2)
    
    Returns
    -------
    xdot : 7x1 state derivatives
    '''

    # extract states
    qw = x[2]
    qz = x[3]
    vx = x[4]
    vy = x[5]
    wz = x[6]

    # extract parameters
    m = tht[0]
    rhox = tht[1]
    rhoy = tht[2]
    Jzz = tht[3]

    # rotation matrix
    R = rot(qw, qz)

    # thruster allocation
    F = mixer(u, R)

    # dynamics
    rho_B = cs.vertcat(rhox, rhoy)
    rho_I = R*rho_B
    M = cs.vertcat(
        cs.horzcat(m, 0, -m*rho_I[1]),
        cs.horzcat(0, m, m*rho_I[0]),
        cs.horzcat(-m*rho_I[1], m*rho_I[0], Jzz)
    )
    C = cs.vertcat(
        -m*rho_I[0]*wz**2,
        -m*rho_I[1]*wz**2,
        0
    )
    Vd = cs.inv(M)@(F-C)

    # kinematics
    v = cs.vertcat(vx, vy)
    pd = v
    q = cs.vertcat(qw, qz)
    Omg = cs.vertcat(
        cs.horzcat(0, -wz),
        cs.horzcat(wz, 0)
    )
    qd = 1/2*Omg@q

    xdot = cs.vertcat(pd, qd, Vd)

    return xdot
    
def x_error(x, xr):
    '''
    Computes the state error
    
    Parameters
    ----------
    x : 7x1 state vector
    xr : 7x1 reference vector

    Returns
    -------
    e : 6x1 error vector
    '''

    # extract states
    p = cs.vertcat(x[0], x[1])
    q = cs.vertcat(x[2], x[3])
    V = cs.vertcat(x[4], x[5], x[6])

    # extract reference states
    pr = cs.vertcat(xr[0], xr[1])
    qr = cs.vertcat(xr[2], xr[3])
    Vr = cs.vertcat(xr[4], xr[5], xr[6])    

    # compute errors
    ep = pr - p
    eq = 1 - (q.T@qr)**2
    eV = Vr - V
    e = cs.vertcat(ep, eq, eV)

    return e

# real mass properties
m = 16.8
Jzz = .1594
rho_x = 0
rho_y = .068
params = np.vstack([m, rho_x, rho_y, Jzz])
u_max = 1.5
r = .12

# initial/commanded states
px = 0
py = 0
theta = 1*np.pi/2
qw = np.cos(theta/2)
qz = np.sin(theta/2)
vx = 0
vy = 0
omg = 0
x0 = np.vstack([px, py, qw, qz, vx, vy, omg])
phi = np.zeros([7,4])
F = np.eye(4)
px_r = 0
py_r = 0
theta_r = 0
qw_r = np.cos(theta_r/2)
qz_r = np.sin(theta_r/2)
vx_r = 0
vy_r = 0
omg_r = 0
xr = np.vstack([px_r, py_r, qw_r, qz_r, vx_r, vy_r, omg_r])

# optimization parameters
h = 20
dt = .5
Q = np.diag([5e1, 5e1, 8e1, 1e1, 1e1, 1e1])
R = 1e-1*np.eye(4)
#l = np.vstack([100, 100, 100, 100])
sig = .01*np.eye(7)

lam_min_vec = np.array([0.1, 0.1, 0.1, 0.1])
lam_max_vec = np.array([10.0, 10.0, 10.0, 10.0])

import math
def lambda_schedule(k, dt, lam_min=1.0, lam_max=10.0, gamma=0.2):
    """
    Exponential decay:
        lambda(0)   = lam_max
        lambda(t→∞) = lam_min

    All arguments are plain Python floats or numpy scalars.
    Returns a plain float.
    """
    t = k * dt
    lam = lam_min + (lam_max - lam_min) * math.exp(-gamma * t)
    # optional clip
    if lam < lam_min:
        lam = lam_min
    elif lam > lam_max:
        lam = lam_max
    return lam


# vector sizes
nx = 7 # number of states
nu = 4 # number of control variables
ntht = 4 # number of parameters
nd = nx*(h+1) + nu*(h) # number of decision variables
ng = nx*(h+1)

# initialize decision variables
X = cs.SX.sym('X', nx, h+1)
U = cs.SX.sym('U', nu, h)

# compute partial derivatives of the system dynamics with respect to the state
x = cs.SX.sym('x', nx, 1)
u = cs.SX.sym('u', nu, 1)
tht = cs.SX.sym('tht', ntht, 1)
Jx = cs.jacobian(x_dot(x,u,tht), x)
Jx_k = cs.Function('Jx', [x,u,tht], [Jx])

Jtht = cs.jacobian(x_dot(x,u,tht), tht)
Jtht_k = cs.Function('Jtht', [x,u,tht], [Jtht])

# initialize cost function and constraints
cost = 0 # accumulated cost
g = [] # list of constraints

# initial condition constraint: X0 = current state
X0 = cs.SX.sym('X0', 7, 1)
g.append(X[:,0] - X0)

# loop over time horizon to build cost function and constraints
for k in range(h):

    # compute error
    e = x_error(X[:,k], xr)

    # compute fisher information matrix
    F += phi.T@cs.inv(sig)@phi
    
    # computee the current choice lambda based on decay rate gamma
    lam_k = lambda_schedule(k, dt, lam_min=1.0, lam_max=500.0, gamma=0.2)
    # running cost
    cost += e.T@Q@e + U[:,k].T@R@U[:,k]

    # dynamics constraint
    X_next = X[:,k] + dt*x_dot(X[:,k],U[:,k],params)
    g.append(X[:,k+1] - X_next)

    # compute new phi
    phi = phi + dt*(Jx_k(X[:,k],U[:,k],params)@phi + Jtht_k(X[:,k],U[:,k],params))

# terminal cost
e = x_error(X[:,h], xr)
cost += e.T@Q@e + lam_k*cs.trace(cs.inv(F))

# assemble decision and constraint vectors
opt_vars = cs.vertcat(cs.reshape(X, -1, 1), cs.reshape(U, -1, 1))
g_concat = cs.vertcat(*g)

# define NLP problem
nlp_problem = {
    'f': cost, # cost function
    'x': opt_vars, # decision variables
    'g': g_concat, # constraints
    'p': X0 # initial condition
}

# create NLP solver using IPOPT
opts = {'ipopt.print_level': 0, 'print_time': 0}
solver = cs.nlpsol('solver', 'ipopt', nlp_problem, opts)

# initialize bounds
lbx = -np.inf*np.ones(nd)
ubx = np.inf*np.ones(nd)
lbg = np.zeros(ng)
ubg = np.zeros(ng)

# set bounds
for i in range(h):
    for j in range(nu):
        idx = nx*(h+1) + i*nu + j
        lbx[idx] = -u_max
        ubx[idx] = u_max

# solve trajectory
sol = solver(p=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
sol_opt = sol['x'].full().flatten()

# extract optimal trajectory
px_opt = []
py_opt = []
qw_opt = []
qz_opt = []
vx_opt = []
vy_opt = []
wz_opt = []
u_opt = []
t_opt = []

for i in range(h):
    px_opt.append(sol_opt[i*nx])
    py_opt.append(sol_opt[i*nx+1])
    qw_opt.append(sol_opt[i*nx+2])
    qz_opt.append(sol_opt[i*nx+3])
    vx_opt.append(sol_opt[i*nx+4])
    vy_opt.append(sol_opt[i*nx+5])
    wz_opt.append(sol_opt[i*nx+6])
    u_opt.append(sol_opt[nx*(h+1)+nu*i:nx*(h+1)+nu*(i+1)])
    t_opt.append(i*dt)

fig, axs = plt.subplots(3,2)
axs[0,0].plot(t_opt, px_opt)
axs[0,0].set_ylabel('x position')
axs[1,0].plot(t_opt, py_opt)
axs[1,0].set_ylabel('y position')
axs[2,0].plot(t_opt, qz_opt)
axs[2,0].set_ylabel('z quaternion')
axs[0,1].plot(t_opt, vx_opt)
axs[0,1].set_ylabel('x velocity')
axs[1,1].plot(t_opt, vy_opt)
axs[1,1].set_ylabel('y velocity')
axs[2,1].plot(t_opt, wz_opt)
axs[2,1].set_ylabel('z angular velocity')
plt.show()

b = np.zeros([3*h, 1])
A = np.zeros([3*h, 4])

# Execute the trajectory and estimate the mass properties
x_hat = np.array(x0)
for k in range(h):
    acc = x_dot(x_hat, u_opt[k], params)

    # Assemble the measurement matrix for this time step
    F = mixer(u_opt[k], rot(x_hat[2], x_hat[3]))

    b[3*k,0] = acc[4]
    b[3*k+1,0] = acc[5]

    A[3*k,0] = F[0]
    A[3*k+1,0] = F[1]
    A[3*k+2,0] = F[2]
    A[3*k,1] = x_hat[6]**2
    A[3*k+1,1] = acc[6]
    A[3*k+2,1] = -acc[5]
    A[3*k,2] = acc[6]
    A[3*k+1,2] = x_hat[6]**2
    A[3*k+2,2] = acc[4]
    A[3*k+2,3] = -acc[6]

    # Propogate the trajectory
    x_hat = x_hat + acc*dt

# estimate using least-squares pseudoinverse
b = np.array(b)
A = np.array(A)
tht_hat = np.linalg.inv(A.T@A)@A.T@b
print(1/tht_hat[0])
print(tht_hat[1])
print(tht_hat[2])
print(tht_hat[3]/tht_hat[0])
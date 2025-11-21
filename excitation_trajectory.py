import numpy as np
import casadi as cs

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
    R = cs.vertcat(
        cs.horzcat(qw**2-qz**2, -2*qw*qz),
        cs.horzcat(2*qw*qz, qw**2-qz**2)
    )

    # thruster allocation
    f_B = cs.vertcat(
        u[0]+u[2],
        u[1]+u[3]
    )
    f_I = R@f_B
    tau = r*(-u[0]-u[1]+u[2]+u[3])
    F = cs.vertcat(f_I, tau)

    # dynamics
    rho_B = cs.vertcat(rhox, rhoy)
    rho_I = R*rho_B
    M = cs.vertcat(
        cs.horzcat(m, 0, -m*rho_I[1]),
        cs.horzcat(0, m, m*rho_I[0]),
        cs.horzcat(-m*rho_I[1], m*rho_I[0], Jzz)
    )
    C = cs.vertcat(
        -rho_I[0]*wz**2,
        -rho_I[1]*wz**2,
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
    qd = .5*Omg@q

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
    eq = 1 - (qr.T@q)**2
    eV = Vr - V
    e = cs.vertcat(ep, eq, eV)

    return e

# floatbot parameters
m = 16.8
Jzz = .1594
u_max = 1.5
r = .12
rho_x = 0
rho_y = 0
params = np.vstack([m, rho_x, rho_y, Jzz])

# initial/commanded states
px = 0
py = 0
theta = 0
qw = np.cos(theta/2)
qz = np.sin(theta/2)
vx = 0
vy = 0
omg = 0
x0 = np.array([px, py, qw, qz, vx, vy, omg])
phi = np.ones([7,4])
px_r = 0
py_r = 0
theta_r = 0
qw_r = np.cos(theta_r/2)
qz_r = np.sin(theta_r/2)
vx_r = 0
vy_r = 0
omg_r = 1
xr = np.array([px_r, py_r, qw_r, qz_r, vx_r, vy_r, omg_r])

# optimization parameters
h = 20
dt = .1
Q = np.diag([5e1, 5e1, 8e3, 1e1, 1e1, 1e1])
R = 1e-1*np.eye(4)
l = 1
sig = .01*np.eye(7)

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
    F = phi.T@cs.inv(sig)@phi

    # running cost
    cost += e.T@Q@e + U[:,k].T@R@U[:,k] + l*cs.trace(cs.inv(F))

    # dynamics constraint
    X_next = X[:,k] + dt*x_dot(X[:,k],U[:,k],params)
    g.append(X[:,k+1] - X_next)

    # compute new phi
    phi = phi + dt*(Jx_k(X[:,k],U[:,k],params)@phi + Jtht_k(X[:,k],U[:,k],params))

# terminal cost
e = x_error(X[:,h], xr)
cost += e.T@Q@e

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

sol = solver(p=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
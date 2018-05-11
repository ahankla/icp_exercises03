import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------
#         Exercise 1: RK4
# ------------------------------------

def rk4(f, x0, y0, x1, n):
    """From RosettaCode.org
    f: y', i.e. the function to integrate. Must be function of x and y, f(x,y).
    x0: starting x position (float), used to calculate time step
    y0: initial y/derivative value (float)
    x1: ending x position (float), used to calculate time step
    n: number of steps (int), used to calcualte time step
    """
    vx = [0] * (n+1)
    vy = [0] * (n+1)
    h = (x1 - x0) / float(n)
    vx[0] = x = x0
    vy[0] = y = y0
    for i in range(1, n + 1):
        k1 = h * f(x, y)
        k2 = h * f(x + 0.5*h, y + 0.5*k1)
        k3 = h * f(x + 0.5*h, y + 0.5*k2)
        k4 = h * f(x + h, y + k3)
        vx[i] = x = x0 + i*h
        vy[i] = y = y + (k1 + k2 + k2 + k3 + k3 + k4) / 6
    return vx, vy


def fun(x, y):
    """function to integrate, i.e. y'= ...
    x: independent variable (float)"""
    return -y

# nlist = [1, 10, 100]; legstr=[]
# for n in nlist:
#     vx, vy = rk4(fun, 0, 1, 10, n)
#     for x, y in list(zip(vx, vy))[::10]:
#         print("%4.1f %10.5f %+12.4e" % (x, y, y - np.exp(-x)))
#
#     vx = np.array(vx)
#     plt.semilogy(vx, vy)
#
#     plt.xlabel("x (or t)")
#     plt.ylabel("y(x) or x(t)")
#     legstr.append("RK 4 with n = "+str(n))
#
# plt.semilogy(vx, np.exp(-vx), linestyle=':') #highest number of steps
# legstr.append("Analytic")
# plt.legend(legstr)
# plt.tight_layout()
# plt.show()


# -------------------------------------
#          Problem 2, part a
# -------------------------------------

def rk4b3(xdot, vdot, x0, v0, m, h, n, distances=False):
    """ Runge-Kutta for 3-body problem. 
    
    Differs from rk4, by integrating each left-hand side
    before moving on to next time step.
    
    rdot: evolution of r
    vdot: evolution of v
    x0: a 3x2 np array -- body1 x/y, body2 x/y, body3 x/y
    v0: a 3x2 np array -- body1 vx/vy, ...
    m: a 3x1 np array with masses.
    h and n: scalars. n an integer, h float
    """
    
    # Initialize Arrays [n+1 time steps, 2 coords (x, y), 3 bodies (0, 1, 2)]
    xt = np.zeros((n+1, 2, 3)) 
    vt = np.zeros((n+1, 2, 3))
    rt = np.zeros((n+1, 2, 3))
    pet = np.zeros((n+1, 1, 3))
    ket = np.zeros((n+1, 1, 3))

    # Warnings
    if x0.shape != (2,3):
        print("Error! Please enter the correct dimensions for initial positions."
              "Current shape is " + x0.shape + ", need (2,3)."
              "Exiting...")
        return 0
    if v0.shape != (2,3):
        print("Error! Please enter the correct dimensions for initial velocities."
              "Current shape is " + v0.shape + ", need (2,3)."
              "Exiting...")
        return 0

    # Timestep t=0: time(0) = initial values
    xi = x0
    vi = v0
    ri = get_vector_distance(x0)
    pei = get_potential_energy(m, ri)
    kei = get_kinetic_energy(m, ri, vi)

    # Assign initial values
    xt[0,:,:] = xi
    vt[0,:,:] = vi
    rt[0,:,:] = ri
    pet[0,:,:] = pei
    ket[0,:,:] = kei

    # Time evolution!
    for i in range(1, n + 1):

        # Calculate coefficients: v(l), x(k)
        l1 = h * vdot(t, xi, vi, m, ri)
        k1 = h * xdot(i-1, xi, vi)
        l2 = h * vdot(i-1 + 0.5*h, xi + 0.5*k1, vi + 0.5*l1, m, ri)
        k2 = h * xdot(i-1 + 0.5*h, xi + 0.5*k1, vi + 0.5*l1)
        l3 = h * vdot(i-1 + 0.5*h, xi + 0.5*k2, vi + 0.5*l2, m, ri)
        k3 = h * xdot(i-1 + 0.5*h, xi + 0.5*k2, vi + 0.5*l2)
        l4 = h * vdot(i-1 + h, xi + k3, vi + l3, m, ri)
        k4 = h * xdot(i-1 + h, xi + k3, vi + l3)

        # Evolve: x(t+1)=x(t)+1/6(l1 + 2*l2 + 2*l3 + l4)
        xi = xi + (k1 + 2*k2 + 2*k3 + k4) / 6.
        vi = vi + (l1 + 2*l2 + 2*l3 + l4) / 6.
        ri = get_vector_distance(xi)
        pei = get_potential_energy(m, ri)
        kei = get_kinetic_energy(m, ri, vi)
    
        # Assign to full list
        xt[i,:,:] = xi
        vt[i,:,:] = vi
        rt[i,:,:] = ri
        pet[0,:,:] = pei
        ket[0,:,:] = kei

    if not distances:
        return xt, vt
    else: 
        return xt, vt, rt


def mag(x):
    """ returns sum of squared matrix """
    # https://stackoverflow.com/questions/9171158/how-do-you-get-the-magnitude-of-a-vector-in-numpy
    return np.sum(x.dot(x))

def norm(x):
    """ returns sum of squared matrix """
    # https://stackoverflow.com/questions/9171158/how-do-you-get-the-magnitude-of-a-vector-in-numpy
    return np.sqrt(x.dot(x))

def get_vector_distance(x):
    """ returns r distances """
    
    # rij are distances (vector of (2,) shape np.arrays)
    r12 = x[:, 1] - x[:, 0]
    r23 = x[:, 2] - x[:, 1]
    r31 = x[:, 0] - x[:, 2]

    return np.stack([r12, r23, r31], axis=1)

def vdot(t, x, v, m, r):
    """ to pass to rk4b3. Complicated gravity.

    r is (2,3) np.array with the first index being coordinate x/y and the second index being the body label.
    return (2,3) np.array
    """
    # Note: C vs Fortran naming convention 0=1, 1=2, 2=3
    # unstack axis=1: rij are distances (vector of (2,) shape np.arrays)
    r12, r23, r31 = r[:,0], r[:,1], r[:,2]

    # rijm are magnitude of distances (scalar)
    r12m = mag(r12)
    r23m = mag(r23)
    r31m = mag(r31)

    # np.arrays of shape (2,)
    # ??????
    a12 = (m[1]*r12)/(r12m**1.5) - (m[2]*r31)/(r31m**1.5) 
    a23 = (m[2]*r23)/(r23m**1.5) - (m[0]*r12)/(r12m**1.5)
    a31 = (m[0]*r31)/(r31m**1.5) - (m[1]*r23)/(r23m**1.5)

    return np.stack([a12, a23, a31], axis=1)


def xdot(t, x, v):
    """ PLACEHOLDER.  xdot = v.  v is 2x3 np.array of velocities. """
    return v

def get_potential_energy(m, r):
    """ Newtonian rescaled potential energy: m1*m2/r12 (no Gravitational constant) """

    # Get: Potential Energies
    PE12 = m[0]*m[1] / norm(r[:,0])
    PE23 = m[1]*m[2] / norm(r[:,1])
    PE31 = m[2]*m[0] / norm(r[:,2])

    # Return sum
    return np.array([PE12, PE23, PE31])

def get_kinetic_energy(m, r, v):
    """ Classical kinetic energy: 1/2*m*v^2. v=norm(v)"""
    return np.array([0.5*m[i]*norm(v[:,i])**2 for i in range(len(m))])

# -----------------------
#     Part a
# -----------------------
# Set: Initial Positions
x1 = -0.97000436; y1 = 0.24308753; vx1 = -0.46620368; vy1 = -0.43236573
x2 = 0.97000436; y2 = -0.24308753; vx2 = -0.46620368; vy2 = -0.43236573
x3 = 0.; y3 = 0.; vx3 = 0.93240737; vy3 = 0.86473146

# Set: Masses
m1 = 1.; m2 = 1.; m3 = 1.
m = np.array([m1, m2, m3])

# Scalars -> Vector
x0 = np.array(np.stack([[x1, y1], [x2, y2], [x3, y3]], axis=1))
v0 = np.array(np.stack([[vx1, vy1], [vx2, vy2], [vx3, vy3]], axis=1))

# Evolve over time
dt = 0.001
t = 2.2
xt, vt = rk4b3(xdot, vdot, x0, v0, m, dt, int(t/dt))

# Visualize
f = 1
plt.figure(f)
f += 1
# Plot Path
plt.plot(xt[:, 0, 0], xt[:, 1, 0])
plt.plot(xt[:, 0, 1], xt[:, 1, 1])
plt.plot(xt[:, 0, 2], xt[:, 1, 2])
# Plot Initial Position
plt.plot(xt[0, 0, 0], xt[0, 1, 0], color="C0", marker="*")
plt.plot(xt[0, 0, 1], xt[0, 1, 1], color="C1", marker="*")
plt.plot(xt[0, 0, 2], xt[0, 1, 2], color="C2", marker="*")
# Labeling
plt.xlabel("x")
plt.ylabel("y")
plt.title("Step size: {} for total time: {}".format(dt, t))
plt.legend(["Body 1", "Body 2", "Body 3"])
#plt.savefig("exercise03_0_stepsize{}_time{}.pdf".format(
#    str(dt).replace(".",""), t))



# -----------------------
#     Part b
# -----------------------
# Set: Initial Positions
# m1 opposes l1 = 3, m2 opposes l2 = 4, m3 opposes l3 = 5.
# x1 = (0,4), x2 = (3,0), x3 = (0,0)
# Center of Mass: sum(m_i*x_i) = sum(m_i) x_com. x_com = (1,1)
# Adjust positions: x1 = (-1,3), x2 = (2,-1), x3 = (-1,-1), x_com = (0,0)
x1, y1 = [-1.,  3.]
x2, y2 = [ 2., -1.]
x3, y3 = [-1., -1.]

# Set: Velocities
vx1 = 0.; vy1 = 0.
vx2 = 0.; vy2 = 0.
vx3 = 0.; vy3 = 0.

# Set: Masses
m1 = 3.; m2 = 4.; m3 = 5.
m = np.array([m1, m2, m3])

# Scalars -> Vectors
x0 = np.array(np.stack([[x1, y1], [x2, y2], [x3, y3]], axis=1))
v0 = np.array(np.stack([[vx1, vy1], [vx2, vy2], [vx3, vy3]], axis=1))

# Evolve over time
dt = 0.001
t = 25
xt, vt, rt = rk4b3(xdot, vdot, x0, v0, m, dt, int(t/dt), distances=True)

## (i) Visualize Trajectories 
plt.figure(f); f += 1
# Plot Path
plt.plot(xt[:, 0, 0], xt[:, 1, 0])
plt.plot(xt[:, 0, 1], xt[:, 1, 1])
plt.plot(xt[:, 0, 2], xt[:, 1, 2])
# Plot Initial Position
plt.plot(xt[0, 0, 0], xt[0, 1, 0], color="C0", marker="*")
plt.plot(xt[0, 0, 1], xt[0, 1, 1], color="C1", marker="*")
plt.plot(xt[0, 0, 2], xt[0, 1, 2], color="C2", marker="*")
# Labeling
plt.xlabel("x")
plt.ylabel("y")
plt.title("Step size: {} for total time: {}".format(dt, t))
plt.legend(["Body 1", "Body 2", "Body 3"])
plt.savefig("exercise03_2_stepsize{}_time{}.pdf".format(
    str(dt).replace(".",""), t))

## (ii) Visualize mutual distances of the three bodies in logarithmic scale
# Take absolute value since they're distances
rt = np.abs(rt)
# Plot
plt.figure(f)
f += 1
# Plot Path
plt.plot(rt[:, 0, 0], xt[:, 1, 0])
plt.plot(rt[:, 0, 1], xt[:, 1, 1])
plt.plot(rt[:, 0, 2], xt[:, 1, 2])
# Plot Initial Position
plt.plot(rt[0, 0, 0], xt[0, 1, 0], color="C0", marker="*")
plt.plot(rt[0, 0, 1], xt[0, 1, 1], color="C1", marker="*")
plt.plot(rt[0, 0, 2], xt[0, 1, 2], color="C2", marker="*")
# Labeling
plt.xlabel("x")
plt.ylabel("y")
plt.title("Distances\nStep size: {} for total time: {}".format(dt, t))
plt.legend(["Body 1", "Body 2", "Body 3"])
# Log Scale
ax = plt.gca()
ax.set_xscale("log", nonposx='clip')
ax.set_yscale("log", nonposy='clip')
# Save
plt.savefig("exercise03_3_stepsize{}_time{}.pdf".format(
    str(dt).replace(".",""), t))

# (iii) Visualize ERROR OF TOTAL ENERGY of the system in logarithmic scaling 
# Get: total energy at t=0
KE_0 = 0
PE_0 = 0
TE_0 = PE_0 + KE_0

# Get: total energy at t=i

# Get: error in total energy = abs(TE(t=i)-TE(t=0))


plt.show()

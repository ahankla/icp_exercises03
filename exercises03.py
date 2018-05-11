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

def rk4b3(xdot, vdot, x0, v0, m, h, n):
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
    
    # Shape: n+1 time steps, 2 coords (x, y), 3 bodies (0, 1, 2)
    xt = np.zeros((n+1, 2, 3)); 
    vt = np.zeros((n+1, 2, 3));

    if x0.shape != (2,3):
        print("Error! Please enter the correct dimensions for initial positions."
              "Current shape is " + x0.shape+", need (2,3)."
              "Exiting...")
        return 0
    if v0.shape != (2,3):
        print("Error! Please enter the correct dimensions for initial velocities."
              "Current shape is " +v0.shape+", need (2,3)."
              "Exiting...")
        return 0

    xt[0, :, :] = x0
    vt[0, :, :] = v0

    # Time evolution!
    for i in range(1, n + 1):
        t = i - 1
        # Note that vdot, rdot should return (2,3) arrays!!
        vi = vt[t, :, :] #i-1 is the time of the previous step
        xi = xt[t, :, :]

        # Calculate coefficients
        l1 = h * vdot(t, xi, vi, m)
        k1 = h * xdot(i-1, xi, vi)
        l2 = h * vdot(i-1 + 0.5*h, xi + 0.5*k1, vi + 0.5*l1, m)
        k2 = h * xdot(i-1 + 0.5*h, xi + 0.5*k1, vi + 0.5*l1)
        l3 = h * vdot(i-1 + 0.5*h, xi + 0.5*k2, vi + 0.5*l2, m)
        k3 = h * xdot(i-1 + 0.5*h, xi + 0.5*k2, vi + 0.5*l2)
        l4 = h * vdot(i-1 + h, xi + k3, vi + l3, m)
        k4 = h * xdot(i-1 + h, xi + k3, vi + l3)

        # Evolve and assign to full list: x(t+1)=x(t)+1/6(l1 + 2*l2 + 2*l3 + l4)
        vinew = vi + (l1 + 2*l2 + 2*l3 + l4) / 6.
        vt[i, :, :] = vinew
        xinew = xi + (k1 + 2*k2 + 2*k3 + k4) / 6.
        xt[i, :, :] = xinew

    return xt, vt



def mag2(x):
    """x is 1x2 np.array. Return scalar"""
    if x.shape != (2,):
        print("Vector shape is " + x.shape + " but should be (2,)!! Exiting...")
        return 0
    return x[0]**2 + x[1]**2


def vdot(t, x, v, m):
    """to pass to rk4b3. Complicated gravity.

    r is (2,3) np.array with the first index being coordinate x/y and the second index being the body label.
    return (2,3) np.array
    """
    r12 = x[:, 1] - x[:, 0]; r12m = mag2(r12) # rij are (2,) shape np.arrays. rijm are scalars
    r23 = x[:, 2] - x[:, 1]; r23m = mag2(r23)
    r31 = x[:, 0] - x[:, 2]; r31m = mag2(r31)
    a12 = m[1]*r12/r12m**1.5 - m[2]*r31/r31m**1.5 # np.arrays of shape (2,)
    a23 = m[2]*r23/r23m**1.5 - m[0]*r12/r12m**1.5
    a31 = m[0]*r31/r31m**1.5 - m[1]*r23/r23m**1.5
    return np.stack([a12, a23, a31], axis=1)


def xdot(t, x, v):
    """xdot = v. v is 2x3 np.array of velocities."""
    return v


# -----------------------
#     Part a
# -----------------------
# Set Initial Positions
x1 = -0.97000436; y1 = 0.24308753; vx1 = -0.46620368; vy1 = -0.43236573
x2 = 0.97000436; y2 = -0.24308753; vx2 = -0.46620368; vy2 = -0.43236573
x3 = 0.; y3 = 0.; vx3 = 0.93240737; vy3 = 0.86473146

# Set Masses
m1 = 1.; m2 = 1.; m3 = 1.;
m = np.array([m1, m2, m3])

# Scalars -> Vector
x0 = np.array(np.stack([[x1, y1], [x2, y2], [x3, y3]], axis=1))
v0 = np.array(np.stack([[vx1, vy1], [vx2, vy2], [vx3, vy3]], axis=1))

# Evolve over time
dt = 0.001
xt, vt = rk4b3(xdot, vdot, x0, v0, m, dt, int(2/dt))

# Visualize
f = 1
plt.figure(f)
f += 1
plt.plot(xt[:, 0, 0], xt[:, 1, 0])
plt.plot(xt[:, 0, 1], xt[:, 1, 1])
plt.plot(xt[:, 0, 2], xt[:, 1, 2])
plt.plot(xt[0, 0, 0], xt[0, 1, 0], color="C0", marker="*")
plt.plot(xt[0, 0, 1], xt[0, 1, 1], color="C1", marker="*")
plt.plot(xt[0, 0, 2], xt[0, 1, 2], color="C2", marker="*")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Step size: " + str(dt))
plt.legend(["Body 1", "Body 2", "Body 3"])
# plt.savefig("exercise03_0_stepsize001.pdf")


# -----------------------
#     Part b
# -----------------------
### XX Haven't updated positions yet or recorded minimum separation
# Set Initial Positions
x1 = -0.97000436; y1 = 0.24308753; vx1 = 0.; vy1 = 0.
x2 = 0.97000436; y2 = -0.24308753; vx2 = 0.; vy2 = 0.
x3 = 0.; y3 = 0.; vx3 = 0.; vy3 = 0.

# Set Masses
m1 = 3.; m2 = 4.; m3 = 5.
m = np.array([m1, m2, m3])

# Scalars -> Vectors
x0 = np.array(np.stack([[x1, y1], [x2, y2], [x3, y3]], axis=1))
v0 = np.array(np.stack([[vx1, vy1], [vx2, vy2], [vx3, vy3]], axis=1))

# Evolve over time
dt = 0.001
xt, vt = rk4b3(xdot, vdot, x0, v0, m, dt, int(2/dt))

# Visualize
plt.figure(f); f += 1
plt.plot(xt[:, 0, 0], xt[:, 1, 0])
plt.plot(xt[:, 0, 1], xt[:, 1, 1])
plt.plot(xt[:, 0, 2], xt[:, 1, 2])
plt.plot(xt[0, 0, 0], xt[0, 1, 0], color="C0", marker="*")
plt.plot(xt[0, 0, 1], xt[0, 1, 1], color="C1", marker="*")
plt.plot(xt[0, 0, 2], xt[0, 1, 2], color="C2", marker="*")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Step size: " + str(dt))
plt.legend(["Body 1", "Body 2", "Body 3"])
# plt.savefig("exercise03_0_stepsize001.pdf")


plt.show()


################################################################
# This Python Code serves as a basic shell for
# 1D-Advection equation wth smooth intitial function and periodic Boundary
# U_t + aU_x = 0    a - advection speed
# Objective: Apply the Upwind scheme, Lax-Friedrichs scheme and Lax-Wendroff scheme for numerical approximation
# Author: Stephen Naboth
# Date: 06/08/2022
#
'''
Conservative form 
U^(j+1)_n = U^(j)_n - dt/dx(F^(j)_n+1/2 - F^(j)_n-1/2) 



Upwind scheme
-------------
Flux_UP = | aU_L, if a>0,
          | aU_R, if a<0,
For smooth case we see that there is a large amount of numerical dffusion and this effect is
stronger for smaller CFL-numbers
To some degree this can be cured by refining the grid, but the diffusive behaviour is still
clearly visible.


Lax-Friedrichs scheme
----------------------
Flux_LF = 0.5*a*(U_R + U_L) - 0.5*dx/dt(U_R - U_L)
In all test cases the results are similar to the upwind-results, but even more dffusive.


Lax-Wendroff scheme
-------------------
Flux_LW = 0.5*a*(U_R + U_L) - a^2*0.5*dt/dx(U_R - U_L)
The smooth case is handled very well by the Lax-Wendroff method, at both large and
small CFL-numbers.

CFL condition = a*dt/dx



'''

################################################################

# ===============================================================
# Some libraries
# ===============================================================

import matplotlib.pyplot as plt  # Plots
import numpy as np  # Numerics

# Customize the plots
from matplotlib import rc
rc('font', family='serif')
rc('lines', linewidth=1.5)
rc('font', size=16)
plt.rc('legend', **{'fontsize': 12})

# ===============================================================
# Some definitions
# ===============================================================


Nx = 81                       # Number of grid points  ----Also try with 101
xmax = 2.                        # Domain limit to the right
xmin = -2.                       # Domain limit to the left
dx = (xmax-xmin)/(Nx-1)           # Mesh size
x = np.arange(xmin, xmax, dx)       # Discretized mesh
dt = 0.04                         # Time step
t_end = 5.                        # Final time
Nt = int(t_end/dt)                # Number of iterations
t = np.linspace(0., t_end, Nt+1)    # Time vector
c = 0.8                           # Advection speed
# CFL number ==> modify CFL by changing Nx (80, 101)
CFL = c*dt/dx

U = np.exp(-0.5 * (x/0.4)**2)   # Initial solution
Uex = U                           # Exact solution

# Chose your scheme: 1 (upwind), 2 (Lax-Wendroff), 3 (Lax-Friedrichs)
scheme = 1


# ===============================================================
# Temporal loop
# ===============================================================
for n in range(1, len(t)):

    # Solve equation using upwind scheme
    if (scheme == 1):

        Un = U
        if (c > 0.):
            Um = np.roll(Un, 1)

            U = Un - CFL*(Un-Um)
        else:
            Up = np.roll(Un, -1)
            U = Un - CFL*(Up-Un)

    # Solve equation using the centered scheme with/without dissipation
    if (scheme == 2):

        theta = (c*dt/dx)**2
        Un = U
        Um = np.roll(Un, 1)
        Up = np.roll(Un, -1)
        # finite volume schemes in conservative form
        U = Un - 0.5*CFL*(Up-Um) + 0.5*theta*(Up-2*Un+Um)

    # Solve equation using the Lax Friedrichs
    if (scheme == 3):

        theta = (c*dx/dt)**2
        Un = U
        Um = np.roll(Un, 1)
        Up = np.roll(Un, -1)
        U = Un - 0.5*CFL*(Up-Um) + 0.5*theta*(Up-2*Un+Um)

    # ===============================================================
    # Compute exact solution
    # ===============================================================
    d = c*n*dt
    Uex = np.exp(-0.5*(np.mod(x-d+xmax, 4)-xmax)**2/0.4**2)
    errL1 = U - Uex
    errL2 = np.linalg.norm(errL1)

    # ===============================================================
    # Plot solution
    # ===============================================================
    if (scheme == 1):
        if (n == 0):
            fig, ax = plt.subplots(figsize=(5.5, 4))
        plt.clf()
        plt.plot(x, U)
        plt.scatter(x, Uex, marker='o', facecolors='white', color='k')
        plt.gca().legend(('Upwind scheme (CFL='+str(CFL)+')', 'Exact solution'))
        plt.axis([xmin, xmax, 0, 1.4])
        plt.title('t='+str(round(dt*(n+1), 3)), fontsize=16)
        plt.xlabel('x', fontsize=18)
        plt.ylabel('u', fontsize=18)
        plt.subplots_adjust(left=0.2)
        plt.subplots_adjust(bottom=0.18)
        plt.draw()
        plt.pause(0.001)

    if (scheme == 2):
        if (n == 1):
            fig, ax = plt.subplots(figsize=(5.5, 4))
        plt.clf()
        plt.plot(x, U)
        plt.scatter(x, Uex, marker='o', facecolors='white', color='k')
        plt.gca().legend(('Lax-Wendroff ($\\theta$=' +
                          str(round(theta, 3))+', CFL='+str(CFL)+')', 'Exact solution'))
        plt.axis([xmin, xmax, 0, 1.4])
        plt.title('t='+str(round(dt*n, 3)), fontsize=16)
        plt.xlabel('x', fontsize=18)
        plt.ylabel('u', fontsize=18)
        plt.subplots_adjust(left=0.2)
        plt.subplots_adjust(bottom=0.18)
        plt.draw()
        plt.pause(0.001)

    if (scheme == 3):
        if (n == 1):
            fig, ax = plt.subplots(figsize=(5.5, 4))
        plt.clf()
        plt.plot(x, U)
        plt.scatter(x, Uex, marker='o', facecolors='white', color='k')
        plt.gca().legend(('Lax Freidrichs ($\\theta$=' +
                          str(round(theta, 3))+', CFL='+str(CFL)+')', 'Exact solution'))
        plt.axis([xmin, xmax, 0, 1.4])
        plt.title('t='+str(round(dt*n, 3)), fontsize=16)
        plt.xlabel('x', fontsize=18)
        plt.ylabel('u', fontsize=18)
        plt.subplots_adjust(left=0.2)
        plt.subplots_adjust(bottom=0.18)
        plt.draw()
        plt.pause(0.001)

plt.show()
#fig.savefig("figure.pdf", dpi=300)
print('Error L2 = ', errL2)

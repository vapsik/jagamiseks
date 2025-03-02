import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Constants
m1 = 1      # Mass of pendulum 1
m2 = 1     # Mass of pendulum 2
m3 = 0.6      # Mass of the middle point
L = 1.0       # Length of each pendulum
g = 9.81      # Gravitational acceleration

# Define the equations of motion
def equations(y, t):
    theta1, theta2, theta1_dot, theta2_dot = y
    A = np.array([
        [m1 * L**2 + m3 * L**2 / 16, m3 * L**2 / 16],
        [m3 * L**2 / 16, m2 * L**2 + m3 * L**2 / 16]
    ])
    b = np.array([
        -m1 * g * L * theta1 - m3 * g * L / 8 * (theta1 + theta2),
        -m2 * g * L * theta2 - m3 * g * L / 8 * (theta1 + theta2)
    ])
    theta_ddots = np.linalg.solve(A, b)  # Solving the matrix equation
    return [theta1_dot, theta2_dot, theta_ddots[0], theta_ddots[1]]

# Initial conditions: theta1, theta2, theta1_dot, theta2_dot
y0 = [0.3, 0, 0, 0]  # Small initial angle for theta1, theta2 at rest

# Time span for the simulation
t_span = (0, 60)
t_eval = np.linspace(0, 60, 1000)

# Solve the differential equations
solution = odeint(equations, y0, t_eval)

# Extract solutions for theta1, theta2, theta1_dot, and theta2_dot
theta1 = solution.T[0]
theta2 = solution.T[1]
theta1_dot = solution.T[2]
theta2_dot = solution.T[3]

# Plot results
f1 = plt.figure(figsize=(12, 12))
plt.subplot(2, 1, 1)
plt.plot(t_eval, theta1, label=r'$\theta_1$ (1. pendli nurk)')
plt.plot(t_eval, theta2, label=r'$\theta_2$ (2. pendli nurk)')
#plt.plot(solution.t, (theta1 + theta2)/2, label=r'$\dot{\theta}_2$ (keskmise pendli nurk)')
plt.xlabel('Aeg (s)')
plt.ylabel('Nurk (rad)')
plt.legend()
plt.grid()
plt.title(f"Kaksikpendlite süsteem - nurgad\n m1 = {m1} kg, m2 = {m2} kg, mk = {m3} kg; " + r'$\theta_{1,2 - algne}$' + f" = {y0[:2]} rad" )

plt.subplot(2, 1, 2)
plt.plot(t_eval, theta1_dot, label=r'$\dot{\theta}_1$ (1. pendli nurkkiirus)')
plt.plot(t_eval, theta2_dot, label=r'$\dot{\theta}_2$ (2. pendli nurkkiirus)')
#plt.plot(solution.t, (theta1_dot + theta2_dot)/2, label=r'$\dot{\theta}_2$ (keskmise pendli nurkkiirus)')
plt.xlabel('Aeg (s)')
plt.ylabel('Nurkkiirus (rad/s)')
plt.legend()
plt.grid()
plt.title(f"Kaksikpendlite süsteem - nurkkiirused")


def energy(theta, thetadot, m, l = L):
    return m*l*(1 - np.cos(theta))*g + 0.5*m*(l*thetadot)**2


f2 = plt.figure(figsize=(8, 6))
plt.plot(t_eval, energy(theta1, theta1_dot, m1), label=r'1. pendli energia')
plt.plot(t_eval, energy(theta2, theta2_dot, m2), label=r'2. pendli energia')
plt.plot(t_eval, energy((theta1 + theta2)/2, (theta1_dot + theta2_dot)/2, m3, l= L/2), label=r'keskmise pendli energia')
plt.xlabel('Aeg (s)')
plt.ylabel('Energia (J)')
plt.legend()
plt.grid()
plt.title(f"Kaksikpendlite süsteem - koguenargiad\n m1 = {m1} kg, m2 = {m2} kg, mk = {m3} kg; " + r'$\theta_{1,2 - algne}$' + f" = {y0[:2]} rad")


plt.tight_layout()

f1.savefig(f"nurgad_m12k={m1,m2,m3}theta12={y0[0],y0[1]}.png")
f2.savefig(f"energiad={m1, m2, m3}.png")
plt.show()
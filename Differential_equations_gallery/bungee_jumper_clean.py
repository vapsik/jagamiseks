import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

k = 500
m = 50
g = 9.81
b = 2

def DE(t, y):
    x, xdot = y
    d = (k/m)*x if x < 0 else 0
    xddot = -g - b/m * xdot*np.abs(xdot) - d 
    return[xdot, xddot]

y0 = [5, -9]

t_span = (0, 1000)
t_eval = np.linspace(0, 30, 1000)

solution = solve_ivp(DE, t_span = t_span, y0=y0, t_eval=t_eval)
solutionR = solve_ivp(DE, t_span = t_span, y0=y0, t_eval=t_eval, method="Radau")
x = solution.y[0]
xR = solutionR.y[0]
xdot = solution.y[1]
xdotR = solutionR.y[1]
xddot = []

for i in range(len(solution.t)):
    xddot.append(DE(0, [x[i], xdot[i]])[1])
xddotR = []
for i in range(len(solutionR.t)):
    xddotR.append(DE(0, [xR[i], xdotR[i]])[1])


f1 = plt.figure(figsize=(12, 12))
plt.subplot(3, 1, 2)
plt.plot(solution.t, xdot, label = r'$v = \frac{dx}{dt}$ RK45', color = "orange")
plt.plot(solutionR.t, xdotR, label = r'$v = \frac{dx}{dt}$ Radau', color = "blue")
plt.ylabel(r'$v (\frac{m}{s})$')
plt.xlabel("t (s)")
plt.grid()
plt.legend()

plt.subplot(3, 1, 1)
plt.plot(solution.t, x, label = r'$x = x(t) $ RK45')
plt.plot(solutionR.t, xR, label = r'$v = \frac{dx}{dt}$ Radau', color = "blue")
plt.ylabel("x (m)")
plt.xlabel("t (s)")
plt.grid()
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(solution.t, xddot, label = r'$a = \frac{dv}{dt} $ RK45', color = "red")
plt.plot(solutionR.t, xddotR, label = r'$a = \frac{dx}{dt}$ Radau', color = "blue")
plt.ylabel(r'$a (\frac{m}{s^2})$')
plt.xlabel("t (s)")
plt.legend()
plt.grid()
plt.show()
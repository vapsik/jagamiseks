from scipy.integrate import solve_ivp
import numpy as np
import plotly.graph_objects as go
from plotly.offline import plot

import pysindy as ps

# Lorenzi süsteemi parameetrid
sigma = 10  
rho = 28
beta = 8/3
noise= 0.00
# DV süsteem
def lorenz(t, xyz):
    x, y, z = xyz
    xdot = sigma * (y - x) + noise * np.random.randn()
    ydot = x * (rho - z) - y + noise * np.random.randn()
    zdot = x * y - beta * z + noise * np.random.randn()
    return [xdot, ydot, zdot]

# Algtingimused
xyz0 = [1, 1, 1]
t_span = (0, 80) #ajaline kestus 80 sekundit
N = 10000 #nö deltade arv
t_eval = np.linspace(*t_span, N)

# Lahendan numbriliselt kasutades scipy.integrate.solve_ivp funktsiooni
solution = solve_ivp(lorenz, y0=xyz0, t_span=t_span, t_eval=t_eval)
X, Y, Z = solution.y

# fikseerin tuletiste väärtused vastavalt DV süsteemi eeskirjale
Xdot, Ydot, Zdot = np.array([lorenz(t, [x, y, z]) for t, x, y, z in zip(t_eval, X, Y, Z)]).T

# Joonistan Lorenzi süsteemi trajektoori
colors = np.linspace(0, 1, len(X))
fig = go.Figure(data=[go.Scatter3d(
    x=X, y=Y, z=Z,
    mode='lines',
    line=dict(
        width=3,
        color=colors,
        colorscale='Viridis'
    )
)])
fig.update_layout(
    title="Lorenz System",
    paper_bgcolor="black",
    plot_bgcolor="rgba(0,0,0,0)",
    scene=dict(
        xaxis=dict(showbackground=False),
        yaxis=dict(showbackground=False),
        zaxis=dict(showbackground=False)
    ),
)
#fig.write_html("/mnt/c/Users/ander/Desktop/plot.html")
fig.show()
# Plot Lorenz system derivatives
fig2 = go.Figure(data=[go.Scatter3d(
    x=Xdot, y=Ydot, z=Zdot,
    mode='lines',
    line=dict(
        width=3,
        color=colors,
        colorscale='Plasma'
    )
)])
fig2.update_layout(
    title="Lorenz System Derivatives",
    paper_bgcolor="black",
    plot_bgcolor="rgba(0,0,0,0)",
    scene=dict(
        xaxis=dict(showbackground=False),
        yaxis=dict(showbackground=False),
        zaxis=dict(showbackground=False)
    ),
)
fig2.show()

states = np.vstack((X, Y, Z)).T
derivatives = np.vstack((Xdot, Ydot, Zdot)).T

model = ps.SINDy()
model.fit(states, t=t_eval[1]- t_eval[0], x_dot=derivatives)

predicted_derivatives = model.predict(states)

print(t_eval[1]- t_eval[0])

mse = np.mean((predicted_derivatives - derivatives) ** 2)
print("SINDy mudeli ruutkeskmine viga:", mse)

print("Andmete põhjal leitud Lorenzi süsteemi väärtused:")
model.print()
print("XYZ andmepunkte:", states.shape)
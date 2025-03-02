import torch
import torch.nn as nn
from torchdiffeq import odeint
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

class ODEFunc(nn.Module):
    def __init__(self):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32), 
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, t, z):
        x, y = z[:, 0:1], z[:, 1:2]
        y_prime_nn = self.net(torch.cat([x, y], dim=1))
        return torch.cat([torch.ones_like(x), y_prime_nn], dim=1)  # dx/dt = 1, dy/dt = net output

z0 = torch.tensor([[0.1, 2.0]], dtype=torch.float32)  
t_span = torch.linspace(0.1, 5.0, 100, dtype=torch.float32)

def compute_loss(func, z0, t_span):
    z_pred = odeint(func, z0, t_span)
    x_pred, y_pred = z_pred[:, :, 0], z_pred[:, :, 1]
    y_prime = torch.gradient(y_pred, spacing=(t_span,), dim=0)[0]
    ode_residual = y_prime - (x_pred / y_pred - y_pred**2)
    return torch.mean(ode_residual**2)

func = ODEFunc()
optimizer = torch.optim.Adam(func.parameters(), lr=1e-3)

for epoch in range(500):
    optimizer.zero_grad()
    loss = compute_loss(func, z0, t_span)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

z_pred = odeint(func, z0, t_span)
x_pred, y_pred = z_pred[:, :, 0].detach().numpy(), z_pred[:, :, 1].detach().numpy()

# võrdlus päris DV-ga
def scipy_ode(t, y):
    return t / y - y**2

y0 = [2]
t_eval = np.linspace(0.1, 5, 100)
solution = solve_ivp(scipy_ode, t_span=(0.1, 5), y0=y0, t_eval=t_eval, method="RK45")

# Plot results
plt.plot(x_pred, y_pred, label="Neural ODE Prediction")
plt.plot(solution.t, solution.y[0], label="Scipy ODE Solution", linestyle="dashed")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Neural ODE vs Scipy ODE Solution")
plt.legend()
plt.grid()
plt.show()
from pysr import PySRRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame()
with open("takistuspingevool.csv", "r") as f:
    df = pd.read_csv(f)

Y = np.array(df["U(V)"]*df["I(mA)"])[:-1].reshape(-1, 1)
X = np.array(df["R(kOom)"])[:-1].reshape(-1, 1)

plt.legend()
plt.grid()

plt.plot(X,Y)


# Initialize PySRRegressor with a custom operator set
model = PySRRegressor(
    niterations=5,  # You can increase iterations for better results
    binary_operators=["+", "*", "/"],  # Basic arithmetic operators
    unary_operators=[
        "square",  # X^2
        "inv"      # 1/X
    ],
    extra_sympy_mappings={
        "inv": lambda x: 1 / x
    },
    model_selection="best",  # Chooses the best model based on score
    loss="loss(x, y) = abs(x - y)",  # Custom loss (absolute difference)
    verbosity=1
)

# Fit model
model.fit(X, Y)

# Print best equation
print("Best equation:", model)

# Use the model to predict over a range
X_range = np.array(np.arange(X.min(), X.max(), 300))
predictions = np.array(model.predict(X_range))

plt.plot(X_range, predictions, color = "red")

plt.show()
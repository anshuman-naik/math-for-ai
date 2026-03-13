
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from linear_regression_math import normal_equation, predict
from gradient_descent import batch_gradient_descent

# Load dataset
df = pd.read_csv("data.csv")

X = df["Hours"].values
y = df["Score"].values

# Add bias term
X = np.column_stack((np.ones(len(X)), X))

# Compute parameters
beta = normal_equation(X, y)

# Compute parameters using gradient descent
beta_gd, losses = batch_gradient_descent(X, y)

# Normal equation predictions
y_pred_normal = X @ beta

# Gradient descent predictions
y_pred_gd = X @ beta_gd

sorted_idx = np.argsort(df["Hours"])
hours_sorted = df["Hours"].values[sorted_idx]
y_normal_sorted = y_pred_normal[sorted_idx]
y_gd_sorted = y_pred_gd[sorted_idx]

plt.scatter(df["Hours"], df["Score"], label="Data")

plt.plot(hours_sorted, y_normal_sorted, label="Normal Equation", linewidth=2)

plt.plot(hours_sorted, y_gd_sorted, label="Gradient Descent", linestyle="dashed")

plt.legend()
plt.show()

# User prediction
new_hour = float(input("Enter study hours: "))

prediction = beta[0] + beta[1] * new_hour

print("Predicted Score:", prediction)

# Compute parameters using normal equation
beta = normal_equation(X, y)

print("Normal Equation Beta:", beta)
print("Gradient Descent Beta:", beta_gd)

# Loss Plot

plt.plot(losses)

plt.xlabel("Iterations")
plt.ylabel("Loss (MSE)")
plt.title("Gradient Descent Convergence")

plt.show()

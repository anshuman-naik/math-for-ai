
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

# Plot
plt.scatter(df["Hours"], df["Score"], label="Data")

# Normal equation line
plt.plot(df["Hours"], y_pred_normal, label="Normal Equation", linewidth=2)

# Gradient descent line
plt.plot(df["Hours"], y_pred_gd, label="Gradient Descent", linestyle="dashed")

plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.title("Linear Regression Comparison")

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

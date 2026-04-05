"""
Batch Gradient Descent for Linear Regression

This script demonstrates:
- Closed form solution (Normal Equation)
- Batch Gradient Descent optimization
- Loss convergence visualization
- Comparison with NumPy implementation
"""


import numpy as np
import matplotlib.pyplot as plt


# -------------------------------
# 1. Generate Data
# -------------------------------
np.random.seed(42)

x = np.arange(1, 51)
y = 3 * x + np.random.randn(50) * 10   # noisy linear data


# -------------------------------
# 2. Normal Equation (Closed Form)
# -------------------------------
def normal_equation(x, y):
    x_bar = np.mean(x)
    y_bar = np.mean(y)

    num = np.sum((x - x_bar) * (y - y_bar))
    den = np.sum((x - x_bar) ** 2)

    m = num / den
    c = y_bar - m * x_bar

    return m, c


# -------------------------------
# 3. Batch Gradient Descent
# -------------------------------
def batch_gradient_descent(x, y, lr=0.001, epochs=1000):
    m = 0
    c = 0

    n = len(x)
    losses = []

    for epoch in range(epochs):
        y_pred = m * x + c

        # Loss (MSE)
        loss = np.mean((y - y_pred) ** 2)
        losses.append(loss)

        # Gradients
        dm = (-2 / n) * np.sum(x * (y - y_pred))
        dc = (-2 / n) * np.sum(y - y_pred)

        # Update
        m -= lr * dm
        c -= lr * dc

    return m, c, losses


# -------------------------------
# 4. Metrics
# -------------------------------
def compute_metrics(y, y_pred):
    mse = np.mean((y - y_pred) ** 2)

    y_bar = np.mean(y)
    ss_total = np.sum((y - y_bar) ** 2)
    ss_residual = np.sum((y - y_pred) ** 2)

    r2 = 1 - (ss_residual / ss_total)

    return mse, r2


# -------------------------------
# 5. Run Everything
# -------------------------------

# Normal Equation
m_ne, c_ne = normal_equation(x, y)

# Gradient Descent
m_gd, c_gd, losses = batch_gradient_descent(x, y, lr=0.001, epochs=5000)

# Predictions
y_pred_gd = m_gd * x + c_gd

# Metrics
mse, r2 = compute_metrics(y, y_pred_gd)

# Numpy Reference
m_np, c_np = np.polyfit(x, y, 1)


# -------------------------------
# 6. Print Results
# -------------------------------
print("----- RESULTS -----")
print(f"Normal Equation     : m = {m_ne:.4f}, c = {c_ne:.4f}")
print(f"Gradient Descent    : m = {m_gd:.4f}, c = {c_gd:.4f}")
print(f"Numpy Polyfit       : m = {m_np:.4f}, c = {c_np:.4f}")
print(f"MSE                : {mse:.4f}")
print(f"R² Score           : {r2:.4f}")

# Test prediction
test = 60
prediction = m_gd * test + c_gd
print(f"Prediction at x=60 : {prediction:.4f}")


# -------------------------------
# 7. Plot Regression Line
# -------------------------------
new_x = np.linspace(min(x), max(x), 100)
new_y = m_gd * new_x + c_gd

plt.figure()
plt.scatter(x, y, label="Data")
plt.plot(new_x, new_y, label="GD Regression Line")

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Linear Regression (Gradient Descent)")
plt.legend()
plt.grid()


# -------------------------------
# 8. Plot Loss Curve
# -------------------------------
plt.figure()
plt.plot(losses)
plt.title("Loss vs Epochs")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.grid()

plt.show()

if __name__ == "__main__":
    pass


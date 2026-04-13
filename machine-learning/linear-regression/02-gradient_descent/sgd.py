"""
Stochastic Gradient Descent (SGD) for Linear Regression

This script demonstrates:
- SGD optimization using single data point updates
- Effect of noisy updates on convergence
- Comparison with Batch Gradient Descent
"""

import numpy as np
import matplotlib.pyplot as plt


# -------------------------------
# 1. Generate Data
# -------------------------------
np.random.seed(42)

x = np.arange(1, 51)
y = 3 * x + np.random.randn(50) * 10


# -------------------------------
# 2. SGD Implementation
# -------------------------------
def stochastic_gradient_descent(x, y, lr=0.00001, epochs=5000):
    m = 0
    c = 0

    n = len(x)
    losses = []

    for epoch in range(epochs):

        # Shuffle data
        indices = np.random.permutation(n)
        x_shuffled = x[indices]
        y_shuffled = y[indices]

        for i in range(n):
            xi = x_shuffled[i]
            yi = y_shuffled[i]

            # Prediction
            y_pred = m * xi + c

            # Error
            error = yi - y_pred

            # Gradients
            dm = -2 * xi * error
            dc = -2 * error

            # Update
            m -= lr * dm
            c -= lr * dc

        # Track loss after each epoch
        y_full_pred = m * x + c
        loss = np.mean((y - y_full_pred) ** 2)
        losses.append(loss)

    return m, c, losses


# -------------------------------
# 3. Run
# -------------------------------
m_sgd, c_sgd, losses = stochastic_gradient_descent(x, y)

print("----- SGD RESULTS -----")
print(f"m = {m_sgd:.4f}, c = {c_sgd:.4f}")


# -------------------------------
# 4. Plot Regression Line
# -------------------------------
new_x = np.linspace(min(x), max(x), 100)
new_y = m_sgd * new_x + c_sgd

plt.figure()
plt.scatter(x, y, label="Data")
plt.plot(new_x, new_y, label="SGD Regression Line")

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Linear Regression (SGD)")
plt.legend()
plt.grid()


# -------------------------------
# 5. Plot Loss Curve
# -------------------------------
plt.figure()
plt.plot(losses)
plt.title("Loss vs Epochs (SGD)")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.grid()

plt.show()


if __name__ == "__main__":
    pass

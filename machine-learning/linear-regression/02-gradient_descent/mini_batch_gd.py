"""
Mini-Batch Gradient Descent for Linear Regression

This script demonstrates:
- Optimization using small batches of data
- Balance between Batch GD and SGD
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
# 2. Mini-Batch GD
# -------------------------------
def mini_batch_gradient_descent(x, y, lr=0.0001, epochs=1000, batch_size=10):
    m = 0
    c = 0

    n = len(x)
    losses = []

    for epoch in range(epochs):

        # Shuffle
        indices = np.random.permutation(n)
        x_shuffled = x[indices]
        y_shuffled = y[indices]

        # Create batches
        for i in range(0, n, batch_size):
            x_batch = x_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]

            # Predictions
            y_pred = m * x_batch + c

            # Errors
            error = y_batch - y_pred

            # Gradients (averaged over batch)
            dm = (-2 / len(x_batch)) * np.sum(x_batch * error)
            dc = (-2 / len(x_batch)) * np.sum(error)

            # Update
            m -= lr * dm
            c -= lr * dc

        # Track loss after epoch
        y_full_pred = m * x + c
        loss = np.mean((y - y_full_pred) ** 2)
        losses.append(loss)

    return m, c, losses


# -------------------------------
# 3. Run
# -------------------------------
m_mb, c_mb, losses = mini_batch_gradient_descent(x, y, lr=0.0001, epochs=2000, batch_size=10)

print("----- MINI-BATCH RESULTS -----")
print(f"m = {m_mb:.4f}, c = {c_mb:.4f}")


# -------------------------------
# 4. Plot Regression Line
# -------------------------------
new_x = np.linspace(min(x), max(x), 100)
new_y = m_mb * new_x + c_mb

plt.figure()
plt.scatter(x, y, label="Data")
plt.plot(new_x, new_y, label="Mini-Batch GD Line")

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Linear Regression (Mini-Batch GD)")
plt.legend()
plt.grid()


# -------------------------------
# 5. Loss Curve
# -------------------------------
plt.figure()
plt.plot(losses)
plt.title("Loss vs Epochs (Mini-Batch)")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.grid()

plt.show()


if __name__ == "__main__":
    pass

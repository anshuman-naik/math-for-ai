import numpy as np

def batch_gradient_descent(X, y, lr=0.01, epochs=1000):
    m, n = X.shape
    beta = np.zeros(n)

    for _ in range(epochs):
        predictions = X @ beta
        gradient = (1/m) * X.T @ (predictions - y)
        beta = beta - lr * gradient

    return beta

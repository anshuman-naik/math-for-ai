import numpy as np

def batch_gradient_descent(X, y, lr=0.01, epochs=1000):
    m, n = X.shape
    beta = np.zeros(n)

    losses = []

    for _ in range(epochs):

        predictions = X @ beta
        errors = predictions - y

        loss = np.mean(errors ** 2)
        losses.append(loss)

        gradient = (1/m) * X.T @ errors
        beta = beta - lr * gradient

    return beta, losses

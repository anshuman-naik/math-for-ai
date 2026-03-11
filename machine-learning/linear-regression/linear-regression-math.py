import numpy as np

def normal_equation(X, y):
    """
    Compute regression parameters using the Normal Equation
    β = (XᵀX)^(-1)Xᵀy
    """
    beta = np.linalg.inv(X.T @ X) @ X.T @ y
    return beta


def predict(X, beta):
    return X @ beta


def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

"""
Feature Scaling Utilities

Implements:
- Standardization (Z-score normalization)

Used for:
- Improving gradient descent convergence
- Stabilizing optimization
"""

import numpy as np


def standardize(x):
    """
    Standardize a feature vector

    Returns:
    x_scaled, mean, std
    """
    mean = np.mean(x)
    std = np.std(x)

    if std == 0:
        raise ValueError("Standard deviation is zero, cannot scale")

    x_scaled = (x - mean) / std
    return x_scaled, mean, std


def apply_standardization(x, mean, std):
    """
    Apply existing scaling parameters (used in inference)
    """
    return (x - mean) / std


def reverse_standardization(x_scaled, mean, std):
    """
    Convert scaled values back to original scale
    """
    return x_scaled * std + mean

# initial_conditions.py

import numpy as np

def gaussian_bump(points, center=(0.5, 0.5), sigma=0.1):
    """
    Generate a 2D Gaussian bump as initial condition.

    Args:
        points (np.ndarray): Array of shape (N, 3) or (N, 2) with point coordinates.
        center (tuple): Center of the Gaussian bump (x0, y0).
        sigma (float): Standard deviation of the Gaussian.

    Returns:
        np.ndarray: Array of initial values at each point.
    """
    x0, y0 = center
    x = points[:, 0]
    y = points[:, 1]
    return np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

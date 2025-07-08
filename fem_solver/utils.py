import numpy as np
def compute_cfl_time_step(points, cells, velocity, cfl_number=0.5):
    """
    Compute the maximum stable time step based on the CFL condition.
    """
    min_h = np.inf
    for cell in cells:
        vertices = points[cell]
        edge_lengths = [
            np.linalg.norm(vertices[0] - vertices[1]),
            np.linalg.norm(vertices[1] - vertices[2]),
            np.linalg.norm(vertices[2] - vertices[0])
        ]
        h = min(edge_lengths)
        if h < min_h:
            min_h = h
    v_mag = np.linalg.norm(velocity)
    return cfl_number * min_h / v_mag if v_mag > 0 else np.inf

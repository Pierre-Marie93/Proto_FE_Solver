import numpy as np
from scipy.sparse import lil_matrix

def assemble_mass_matrix(points, cells):
    num_points = len(points)
    M = lil_matrix((num_points, num_points))
    for cell in cells:
        vertices = points[cell]
        area = 0.5 * abs(np.linalg.det([[1, *vertices[0][:2]],
                                    [1, *vertices[1][:2]],
                                    [1, *vertices[2][:2]]]))
        for i in range(3):
            for j in range(3):
                M[cell[i], cell[j]] += area / 12.0 if i != j else area / 6.0
    return M

def compute_element_gradients(vertices):
    """Compute gradients of P1 basis functions on a triangle."""
    area = 0.5 * abs(np.linalg.det([[1, *vertices[0][:2]],
                                    [1, *vertices[1][:2]],
                                    [1, *vertices[2][:2]]]))
    grads = []
    for i in range(3):
        grad_phi = np.array([
            vertices[(i+1)%3,1] - vertices[(i+2)%3,1],
            vertices[(i+2)%3,0] - vertices[(i+1)%3,0]
        ]) / (2 * area)
        grads.append(grad_phi)
    return grads, area

def assemble_advection_matrix(points, cells, velocity):
    num_points = len(points)
    A = lil_matrix((num_points, num_points))

    for cell in cells:
        vertices = points[cell]
        grads, area = compute_element_gradients(vertices)

        for i in range(3):
            for j in range(3):
                A[cell[i], cell[j]] += area * np.dot(velocity, grads[j]) / 3.0

    return A

def assemble_supg_advection_matrix(points, cells, velocity):
    num_points = len(points)
    A = assemble_advection_matrix(points, cells, velocity).tolil()
    v_mag = np.linalg.norm(velocity)

    for cell in cells:
        indices = cell
        vertices = points[indices]
        grads, area = compute_element_gradients(vertices)

        h = max([
            np.linalg.norm(vertices[0] - vertices[1]),
            np.linalg.norm(vertices[1] - vertices[2]),
            np.linalg.norm(vertices[2] - vertices[0])
        ])
        tau = h / (2 * v_mag) if v_mag > 0 else 0

        for i in range(3):
            for j in range(3):
                supg_term = tau * area * np.dot(velocity, grads[i]) * np.dot(velocity, grads[j])
                A[indices[i], indices[j]] += supg_term

    return A


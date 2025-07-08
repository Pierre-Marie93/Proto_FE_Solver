import numpy as np
from collections import defaultdict

def extract_boundary_nodes_by_tag(mesh):
    """
    Extracts boundary nodes grouped by physical tag from a meshio mesh.

    Returns:
        dict: A dictionary mapping physical tag to a set of node indices.
    """
    line_cells = mesh.cells_dict.get("line", [])
    line_tags = mesh.cell_data_dict["gmsh:physical"]["line"]
    boundary_nodes = defaultdict(set)

    for i, tag in enumerate(line_tags):
        for node in line_cells[i]:
            boundary_nodes[tag].add(node)

    return boundary_nodes

def apply_dirichlet_from_labels(mesh, b, M, dirichlet_tags, time=0.0, value_function=None):
    """
    Applies Dirichlet boundary conditions using Gmsh physical labels.

    Args:
        mesh (meshio.Mesh): The mesh object.
        b (np.ndarray): The right-hand side vector.
        M (scipy.sparse matrix): The system matrix.
        dirichlet_tags (list of int): List of physical tags for Dirichlet conditions.
        time (float): Current time (for time-dependent BCs).
        value_function (callable): Function (t, x, y) -> value for Dirichlet BC.

    Returns:
        tuple: Updated (b, M)
    """
    boundary_nodes = extract_boundary_nodes_by_tag(mesh)

    dirichlet_nodes = set()
    for tag in dirichlet_tags:
        dirichlet_nodes.update(boundary_nodes.get(tag, set()))

    for node in dirichlet_nodes:
        x, y = mesh.points[node][:2]
        value = value_function(time, x, y) if value_function else 1.0
        b[node] = value
        M[node, :] = 0.0
        M[node, node] = 1.0

    return b, M


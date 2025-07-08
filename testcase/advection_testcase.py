import numpy as np
from testcase.testcaseClass import TestCase
from testcase.initial_conditions import gaussian_bump


# --- Define boundary condition function ---
def dirichlet_bc(t, x, y):
    # return np.sin(np.pi * t) * (1 - y)
    return np.sin(np.pi * t)

# --- Define boundary condition dictionary ---
boundary_conditions = {
    "left":   {"type": "Dirichlet", "tag": 1, "value_fn": dirichlet_bc},
    "bottom": {"type": "Neumann",  "tag": 2},
    "top":    {"type": "Neumann",  "tag": 3},
    "right":  {"type": "Neumann",  "tag": 4}
}

# --- Define geometry function ---
def unit_square_geometry(mesh_size, boundary_conditions):
    import gmsh
    gmsh.model.add("unit_square")

    # Points
    p1 = gmsh.model.geo.addPoint(0, 0, 0, mesh_size)
    p2 = gmsh.model.geo.addPoint(1, 0, 0, mesh_size)
    p3 = gmsh.model.geo.addPoint(1, 1, 0, mesh_size)
    p4 = gmsh.model.geo.addPoint(0, 1, 0, mesh_size)

    # Lines
    lines = {
        "bottom": gmsh.model.geo.addLine(p1, p2),
        "right":  gmsh.model.geo.addLine(p2, p3),
        "top":    gmsh.model.geo.addLine(p3, p4),
        "left":   gmsh.model.geo.addLine(p4, p1)
    }

    # Surface
    cl = gmsh.model.geo.addCurveLoop([lines["bottom"], lines["right"], lines["top"], lines["left"]])
    surface = gmsh.model.geo.addPlaneSurface([cl])

    # Tag boundaries
    for name, bc in boundary_conditions.items():
        gmsh.model.addPhysicalGroup(1, [lines[name]], tag=bc["tag"])
        gmsh.model.setPhysicalName(1, bc["tag"], f"{bc['type']}_{name.capitalize()}")

    gmsh.model.addPhysicalGroup(2, [surface], tag=5)
    gmsh.model.setPhysicalName(2, 5, "Domain")

def create_advection_testcase():
    """
    Creates a test case for advection using a Gaussian bump as the initial condition.
    """
    return TestCase(
        name="Gaussian Advection",
        mesh_size=0.05,
        velocity=(1.0, 0.0),
        initial_condition_fn=lambda pts: gaussian_bump(pts, center=(0.5, 0.5), sigma=0.1),
        boundary_conditions=boundary_conditions,
        geometry_definition_fn=unit_square_geometry,
        cfl_number=0.1,
        total_time=1.0
    )

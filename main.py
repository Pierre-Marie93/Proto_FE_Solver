import numpy as np
from fem_solver.mesh_utils import generate_labeled_mesh, check_triangle_orientations
from fem_solver.assembly import assemble_mass_matrix, assemble_supg_advection_matrix
from fem_solver.utils import compute_cfl_time_step
from fem_solver.boundary import apply_dirichlet_from_labels
from fem_solver.solver import solve_system
from fem_solver.visualization import plot_scalar_field, plot_mesh
from testcase.testcaseClass import TestCase
from testcase.advection_testcase import create_advection_testcase


# --- Run simulation ---
def run_simulation(test_case, plot_interval=5):
    mesh = generate_labeled_mesh(
        mesh_size=test_case.mesh_size,
        geometry_definition_fn=test_case.geometry_definition_fn,
        boundary_conditions=test_case.boundary_conditions
    )
    check_triangle_orientations(mesh)
    plot_mesh(mesh, show_tags=True)
    points = mesh.points
    cells = mesh.cells_dict["triangle"]
    mass_matrix = assemble_mass_matrix(points, cells).tocsr()
    advection_matrix = assemble_supg_advection_matrix(points, cells, test_case.velocity).tocsr()

    U = test_case.initial_condition_fn(points)
    Delta_t = compute_cfl_time_step(points, cells, test_case.velocity, test_case.cfl_number)
    steps = int(test_case.total_time / Delta_t)

    time = 0.0
    for step in range(steps):
        A = mass_matrix - Delta_t * advection_matrix
        b = A.dot(U)
        mass_matrix_copy = mass_matrix.copy()

        dirichlet_tags = [bc["tag"] for bc in test_case.boundary_conditions.values() if bc["type"] == "Dirichlet"]
        value_fn = next((bc["value_fn"] for bc in test_case.boundary_conditions.values()
                         if bc["type"] == "Dirichlet" and "value_fn" in bc), None)

        b, mass_matrix_copy = apply_dirichlet_from_labels(
            mesh, b, mass_matrix_copy,
            dirichlet_tags=dirichlet_tags,
            time=time,
            value_function=value_fn
        )

        U = solve_system(mass_matrix_copy, b)

        if step % plot_interval == 0:
            plot_scalar_field(U, mesh)

        time += Delta_t

    return mesh, U


# --- Main entry point ---
if __name__ == "__main__":
    advection_test_case = create_advection_testcase()
    mesh, U = run_simulation(advection_test_case,plot_interval=50)
    

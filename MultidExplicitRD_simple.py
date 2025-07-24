import sys
import numpy as np
import warnings

from fem_solver.mesh_utils import generate_labeled_mesh, check_triangle_orientations,compute_nodal_P1_integrals
from testcase.advection_testcase import create_testcase as create_advection_testcase
from testcase.canal_testcase import create_testcase as create_canal_testcase
from fem_solver.utils import compute_cfl_time_step
from fem_solver.visualization import plot_scalar_field, plot_mesh
from fem_solver.assembly import compute_element_gradients
from fem_solver.boundary import apply_dirichlet_from_labels, extract_boundary_nodes_by_tag, invert_boundary_node_map


def compute_normals_triangle(verts):
    normals = []
    for i in range(3):
        # Vecteur du côté opposé au sommet i
        # print(f"Vertsices of triangle {i}: {verts}")
        edge = verts[(i+2)%3,:] - verts[(i+1)%3,:]
        # print(f"Edge {i}: {edge}")
        # Normale extérieure (rotation de l'edge de +90°)
        # normal = np.array([edge[1], -edge[0]])
        # # Normale intérieure (rotation de l'edge de -90°)
        normal = np.array([-edge[1], edge[0]])
        if abs(np.dot(normal,edge)) > 1e-10:
            print("Errror: SP != 0 in normals =", np.sum(normal*edge))
            # print("normal:", np.sum(normals,axis=0))
        normals.append(normal)
    normals = np.array(normals)
    return normals



def distributed_residuals(k, u_tri, t_idx):
    phi_T = np.sum(u_tri * k)
    phi_repar_T = 1./3.*np.ones(3) * phi_T
    return phi_repar_T

def assemble_contribution_triangles(points, triangles, u, velocity, mesh=None):
    nVertex = len(points)
    phi_repar_global = np.zeros((len(triangles),3)) 
    for t_idx, tri in enumerate(triangles):
        verts = points[tri]
        u_tri = u[tri]
        normals = compute_normals_triangle(verts)
        k = (1./2.)*(np.array([np.dot(velocity, normals[i]) for i in range(3)]))
        if abs(np.sum(k)) > 1e-10:
            print("Errror: sum(k) != 0 in assemble_contribution_triangles k=", k, "at triangle", t_idx)
            print("normal:", np.sum(normals,axis=0))
        phi_repar_T = distributed_residuals(k, u_tri, t_idx)
        phi_repar_global[t_idx,:] = phi_repar_T 
    return  phi_repar_global
        
def sum_contributions(points,triangles, phi_repar_global):
    evolution = np.zeros(len(points))
    for t_idx, tri in enumerate(triangles):
        for local_i, global_i in enumerate(tri):
            evolution[global_i] += phi_repar_global[t_idx, local_i]
    return evolution

def evolve_values(points, u, Delta_t,evolution,nodal_integrals):
    """
    Update the values of u based on the evolution and beta_sum.
    """
    nVertex = len(points)
    u_new=np.zeros(np.shape(u))
    for i in range(nVertex):
        u_new[i] = u[i] - (Delta_t / nodal_integrals[i]) * evolution[i]
        # u_new[i] = u[i] - Delta_t * evolution[i]
    return u_new

def run_flux_scheme(test_case, plot_interval_time=1.0):
    mesh = generate_labeled_mesh(
        mesh_size=test_case.mesh_size,
        geometry_definition_fn=test_case.geometry_definition_fn,
        boundary_conditions=test_case.boundary_conditions
    )
    plot_mesh(mesh, show_tags=True)
    check_triangle_orientations(mesh)
    points = mesh.points[:,:2]
    # print("Points", np.shape(points), points)
    triangles = mesh.cells_dict["triangle"]
    # print("Triangles", np.shape(triangles), triangles)
    velocity = test_case.velocity
    U = test_case.initial_condition_fn(points)
    U_prev = U.copy()
    Delta_t = 0.001
    steps = int(test_case.total_time / Delta_t)
    plot_interval = max(1, int(plot_interval_time / Delta_t))
    nodal_integrals = compute_nodal_P1_integrals(mesh)

    for step in range(steps):
        contributions = assemble_contribution_triangles(points, triangles, U, velocity, mesh=mesh)
        evolution = sum_contributions(points,triangles, contributions)
        U = evolve_values(points,U, Delta_t, evolution,nodal_integrals)
        if step % plot_interval == 0:
            plot_scalar_field(U, mesh)
            # plot_scalar_field(U-U_prev, mesh)
            U_prev = U.copy()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py [advection|canal]")
        sys.exit(1)
    testcase_type = sys.argv[1].lower()
    if testcase_type == "advection":
        test_case = create_advection_testcase()
    elif testcase_type == "canal":
        test_case = create_canal_testcase()
    else:
        print("Unknown testcase type. Use 'advection' or 'canal'.")
        sys.exit(1)    
    run_flux_scheme(test_case, plot_interval_time=0.1)
    
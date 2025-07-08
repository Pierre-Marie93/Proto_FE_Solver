import gmsh
import meshio

def generate_labeled_mesh(mesh_size, geometry_definition_fn, boundary_conditions):
    import gmsh
    import meshio

    gmsh.initialize()
    geometry_definition_fn(mesh_size, boundary_conditions)
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)
    gmsh.write("rectangle_labeled.msh")
    gmsh.finalize()

    return meshio.read("rectangle_labeled.msh")

def check_triangle_orientations(mesh):
    points = mesh.points
    triangles = mesh.cells_dict.get("triangle", [])

    ccw_count = 0
    cw_count = 0

    for tri in triangles:
        A, B, C = points[tri[0]], points[tri[1]], points[tri[2]]
        x1, y1 = A[0], A[1]
        x2, y2 = B[0], B[1]
        x3, y3 = C[0], C[1]

        # Compute signed area
        signed_area = 0.5 * ((x2 - x1)*(y3 - y1) - (x3 - x1)*(y2 - y1))

        if signed_area > 0:
            ccw_count += 1
        else:
            cw_count += 1

    print(f"Total triangles: {len(triangles)}")
    print(f"Counterclockwise (CCW): {ccw_count}")
    print(f"Clockwise (CW): {cw_count}")


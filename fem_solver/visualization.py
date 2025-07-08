import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

def plot_mesh(mesh, show_tags=False):
    points = mesh.points
    cells = mesh.cells_dict["triangle"]
    plt.figure()
    plt.triplot(points[:, 0], points[:, 1], cells)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('2D Mesh Plot')

    if show_tags and "line" in mesh.cells_dict:
        line_cells = mesh.cells_dict["line"]
        line_tags = mesh.cell_data_dict.get("gmsh:physical", {}).get("line", [])
        for i, edge in enumerate(line_cells):
            tag = line_tags[i] if i < len(line_tags) else None
            x = points[edge, 0].mean()
            y = points[edge, 1].mean()
            plt.text(x, y, str(tag), color='red', fontsize=8, ha='center')

    plt.axis('equal')
    plt.grid(True)
    plt.show()


def plot_scalar_field(U, mesh):
    points = mesh.points
    cells = mesh.cells_dict["triangle"]
    triangulation = Triangulation(points[:, 0], points[:, 1], cells)
    plt.figure()
    plt.tricontourf(triangulation, U, cmap='plasma')
    plt.colorbar(label='Scalar Field')
    plt.title('Scalar Field Color Map on Gmsh Mesh')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

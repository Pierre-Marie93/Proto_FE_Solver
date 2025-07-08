import numpy as np
class TestCase:
    def __init__(self, name, mesh_size, velocity, initial_condition_fn,
                 boundary_conditions, geometry_definition_fn,
                 cfl_number=0.5, total_time=1.0):
        self.name = name
        self.mesh_size = mesh_size
        self.velocity = np.array(velocity)
        self.initial_condition_fn = initial_condition_fn
        self.boundary_conditions = boundary_conditions
        self.geometry_definition_fn = geometry_definition_fn
        self.cfl_number = cfl_number
        self.total_time = total_time

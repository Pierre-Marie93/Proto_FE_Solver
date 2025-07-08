from scipy.sparse.linalg import splu

def solve_system(M, b):
    lu = splu(M)
    return lu.solve(b)

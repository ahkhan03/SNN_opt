"""
Example 3: Linear Program (LP)

Minimize a linear objective subject to linear constraints.
Demonstrates LP solving by setting A = 0 (no quadratic term).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from snn_opt import OptimizationProblem, SNNSolver, SolverConfig

# Problem: minimize b^T x subject to constraints
# The solution should lie at a vertex of the feasible polytope

n = 3

# A = 0 makes this a pure LP
A = np.zeros((n, n))

# Linear cost: prefer negative x1, x2, x3
b = np.array([1.0, 1.0, 1.0])

# Box constraints: -1 <= x_i <= 2 for all i
# Rewritten as: x_i <= 2 and -x_i <= 1
C = np.array([
    [1.0, 0.0, 0.0],   # x1 <= 2
    [0.0, 1.0, 0.0],   # x2 <= 2
    [0.0, 0.0, 1.0],   # x3 <= 2
    [-1.0, 0.0, 0.0],  # -x1 <= 1  =>  x1 >= -1
    [0.0, -1.0, 0.0],  # -x2 <= 1  =>  x2 >= -1
    [0.0, 0.0, -1.0],  # -x3 <= 1  =>  x3 >= -1
])
d = np.array([-2.0, -2.0, -2.0, -1.0, -1.0, -1.0])

# Initial guess at origin
x0 = np.array([0.0, 0.0, 0.0])

print("=" * 60)
print("Example 3: Linear Program")
print("=" * 60)
print(f"Problem: minimize b^T x")
print(f"Cost vector b: {b}")
print(f"Box constraints: -1 <= x_i <= 2 for i=1,2,3")
print(f"Initial guess: {x0}")
print()

# Expected solution: x = [-1, -1, -1] (corner of box minimizing b^T x)
x_expected = np.array([-1.0, -1.0, -1.0])
obj_expected = b @ x_expected
print(f"Expected solution: {x_expected}")
print(f"Expected objective: {obj_expected:.6f}")
print()

problem = OptimizationProblem(A=A, b=b, C=C, d=d)
config = SolverConfig(k0=0.1, k1=0.1, t_end=100)
solver = SNNSolver(problem, config)

print("Solving...")
result = solver.solve(x0, verbose=False)
print()

print(result.summary())

print("\nComparison with expected solution:")
print(f"Solution error: ||x_sol - x_expected|| = {np.linalg.norm(result.final_x - x_expected):.6e}")
print(f"Objective error: {abs(result.final_objective - obj_expected):.6e}")
print()

print("Active constraints at solution:")
g_final = problem.constraint_values(result.final_x)
constraint_names = ["x1 <= 2", "x2 <= 2", "x3 <= 2", "x1 >= -1", "x2 >= -1", "x3 >= -1"]
for i, (g_i, name) in enumerate(zip(g_final, constraint_names)):
    if abs(g_i) < 1e-3:
        print(f"  {name}: ACTIVE (g = {g_i:.6e})")


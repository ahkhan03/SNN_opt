"""
Example 2: 3D Quadratic Program with Multiple Constraints

Minimize ||x||^2 subject to a polytope defined by 4 hyperplanes.
This corresponds to example1.m in the MATLAB implementation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from snn_opt import OptimizationProblem, SNNSolver, SolverConfig

# Problem: minimize (1/2) x^T x subject to 4 linear constraints
# defining a bounded polytope in 3D

n = 3
A = np.eye(n)
b = np.zeros(n)

# Constraints from MATLAB example1.m
C = np.array([
    [1.0, 1.0, 1.0],
    [1.0, -1.0, 1.0],
    [-1.0, -1.0, -1.0],
    [-1.0, 1.0, -1.0]
])
d = np.array([-1.05, -1.05, 0.95, 0.95])

# Initial guess (outside feasible region)
x0 = np.array([1.0, 1.0, 1.0])

print("=" * 60)
print("Example 2: 3D QP with Polytope Constraints")
print("=" * 60)
print(f"Problem dimension: {n}")
print(f"Number of constraints: {C.shape[0]}")
print(f"Initial guess: {x0}")
print(f"Initial objective: {0.5 * x0.T @ A @ x0:.6f}")
print()

# Create problem and solver
problem = OptimizationProblem(A=A, b=b, C=C, d=d)
config = SolverConfig(k0=0.01, k1=0.01, t_end=200)
solver = SNNSolver(problem, config)

# Check initial feasibility
print(f"Initial point feasible: {problem.is_feasible(x0)}")
print(f"Initial max violation: {problem.max_violation(x0):.6e}")
print()

# Solve
print("Solving...")
result = solver.solve(x0, verbose=False)
print()

print(result.summary())

# Analyze solution
print("\nConstraint analysis at solution:")
g_final = problem.constraint_values(result.final_x)
for i, g_i in enumerate(g_final):
    status = "ACTIVE" if abs(g_i) < 1e-3 else "inactive"
    satisfied = "✓" if g_i <= 1e-6 else "✗"
    print(f"  Constraint {i+1}: {g_i:+.6e}  [{status}] {satisfied}")

print(f"\nSolution components:")
for i, x_i in enumerate(result.final_x):
    print(f"  x[{i+1}] = {x_i:.6f}")

# Verify optimality condition (at minimum of ||x||^2, gradient should point outward from feasible region)
grad = problem.gradient(result.final_x)
print(f"\nGradient at solution: {grad}")
print(f"Gradient magnitude: {np.linalg.norm(grad):.6e}")


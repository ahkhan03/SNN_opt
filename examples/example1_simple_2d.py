"""
Example 1: Simple 2D Quadratic Program

Minimize ||x||^2 subject to linear inequality constraints.
This corresponds to example2.m in the MATLAB implementation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from snn_opt import solve_qp

# Problem: minimize (1/2) x^T x subject to:
#   x1 + 2*x2 <= 1
#  -x1 + 3*x2 <= 1

n = 2
A = np.eye(n)
b = np.zeros(n)

# Constraints: C*x + d <= 0
# x1 + 2*x2 <= 1  =>  x1 + 2*x2 - 1 <= 0
# -x1 + 3*x2 <= 1  => -x1 + 3*x2 - 1 <= 0
C = np.array([[1.0, 2.0],
              [-1.0, 3.0]])
d = np.array([-1.0, -1.0])

# Initial guess (infeasible)
x0 = np.array([1.0, 1.0])

# Solve
print("=" * 60)
print("Example 1: Simple 2D QP")
print("=" * 60)
print(f"Problem: minimize ||x||^2")
print(f"Subject to: x1 + 2*x2 <= 1")
print(f"           -x1 + 3*x2 <= 1")
print(f"Initial guess: {x0}")
print()

# Using auto k0 (k0=None) and adaptive projection (default)
result = solve_qp(A, b, C, d, x0, max_iterations=1000, verbose=False)

print(result.summary())
print()
print(f"Trajectory points: {len(result.t)}")
print(f"Time range: [{result.t[0]:.2f}, {result.t[-1]:.2f}]")
print()

# Check constraints at solution
g_final = C @ result.final_x + d
print("Final constraint values (should be <= 0):")
for i, g_i in enumerate(g_final):
    status = "✓" if g_i <= 1e-6 else "✗"
    print(f"  Constraint {i+1}: {g_i:.6e} {status}")
print()

# Print trajectory statistics
print(f"Initial objective: {result.objective_values[0]:.6e}")
print(f"Final objective: {result.final_objective:.6e}")
print(f"Objective reduction: {(result.objective_values[0] - result.final_objective):.6e}")


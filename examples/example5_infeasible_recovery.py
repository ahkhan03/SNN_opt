"""
Example 5: Infeasible Initialization and Recovery

Demonstrates how the solver handles infeasible starting points and 
projects them back to feasibility before optimization.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from snn_opt import OptimizationProblem, SNNSolver, SolverConfig

# Simple 2D problem with tight constraints
n = 2
A = np.eye(n)
b = np.zeros(n)

# Define a diamond-shaped feasible region
# |x1| + |x2| <= 1
# Rewritten as 4 constraints:
#   x1 + x2 <= 1
#  -x1 + x2 <= 1
#   x1 - x2 <= 1
#  -x1 - x2 <= 1
C = np.array([
    [1.0, 1.0],
    [-1.0, 1.0],
    [1.0, -1.0],
    [-1.0, -1.0]
])
d = np.array([-1.0, -1.0, -1.0, -1.0])

print("=" * 60)
print("Example 5: Infeasible Initialization Recovery")
print("=" * 60)
print("Problem: minimize ||x||^2")
print("Feasible region: Diamond shape |x1| + |x2| <= 1")
print()

problem = OptimizationProblem(A=A, b=b, C=C, d=d)
config = SolverConfig(k0=0.05, k1=0.05, t_end=100)

# Test several infeasible starting points
test_points = [
    np.array([2.0, 2.0]),
    np.array([3.0, 0.0]),
    np.array([0.0, -5.0]),
    np.array([-2.0, 2.0]),
]

print(f"{'Initial x':<20} {'Violation':<12} {'Final x':<20} {'Objective':<12} {'Projections':<12}")
print("-" * 88)

for x0 in test_points:
    # Check initial feasibility
    violation = problem.max_violation(x0)
    
    # Solve
    solver = SNNSolver(problem, config)
    result = solver.solve(x0, verbose=False)
    
    # Report
    x0_str = f"[{x0[0]:+.2f}, {x0[1]:+.2f}]"
    xf_str = f"[{result.final_x[0]:+.5f}, {result.final_x[1]:+.5f}]"
    
    print(f"{x0_str:<20} {violation:<12.3e} {xf_str:<20} "
          f"{result.final_objective:<12.6e} {result.n_projections:<12}")

print()

# Detailed analysis for one case
print("Detailed analysis for x0 = [2.0, 2.0]:")
x0 = np.array([2.0, 2.0])
print(f"  Initial point: {x0}")
print(f"  Initial feasibility: {problem.is_feasible(x0)}")
print(f"  Initial constraint values:")
g0 = problem.constraint_values(x0)
constraint_names = ["x1+x2<=1", "-x1+x2<=1", "x1-x2<=1", "-x1-x2<=1"]
for name, g in zip(constraint_names, g0):
    status = "✓ feasible" if g <= 0 else "✗ VIOLATED"
    print(f"    {name}: {g:+.3f} {status}")

print()
solver = SNNSolver(problem, config)
result = solver.solve(x0, verbose=False)

print(f"  After projection phase:")
print(f"    Projections needed: {result.n_projections}")
print(f"  Final solution: {result.final_x}")
print(f"  Final objective: {result.final_objective:.6e}")
print(f"  Final constraint values:")
gf = problem.constraint_values(result.final_x)
for name, g in zip(constraint_names, gf):
    status = "✓" if g <= 1e-6 else "✗"
    active = "(ACTIVE)" if abs(g) < 1e-3 else ""
    print(f"    {name}: {g:+.6e} {status} {active}")

# The optimal solution for this problem is x = [0, 0]
print()
print(f"Expected optimal solution: [0, 0]")
print(f"Solution error: {np.linalg.norm(result.final_x):.6e}")


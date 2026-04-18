"""
Example 4: Warm Starting for Time-Varying Problems

Demonstrates the receding horizon approach where we solve a sequence of 
similar optimization problems, using the previous solution as initialization.
This mimics the manipulator control scenario.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from snn_opt import OptimizationProblem, SNNSolver, SolverConfig
import time

# Simulate a control scenario where the constraint moves over time
n = 2
A = np.eye(n)
b = np.zeros(n)

# Initial constraint: x1 + x2 <= 1
C_base = np.array([[1.0, 1.0]])
d_base = -1.0

print("=" * 60)
print("Example 4: Warm Starting (Receding Horizon)")
print("=" * 60)
print("Simulating time-varying constraint scenario")
print(f"Base problem: minimize ||x||^2")
print(f"Constraint moves: x1 + x2 <= 1 + 0.1*k (k = timestep)")
print()

# Simulation parameters
n_steps = 10
dt = 0.1
solver_config = SolverConfig(k0=0.05, k1=0.05, t_end=50)

# Storage
solutions = []
solve_times = []
iterations = []

# Initial guess
x_current = np.array([2.0, 2.0])

print(f"{'Step':<6} {'d_offset':<10} {'Objective':<12} {'x1':<10} {'x2':<10} {'Time(ms)':<10} {'Projs':<8}")
print("-" * 76)

for k in range(n_steps):
    # Update constraint (move boundary outward)
    d_k = d_base - 0.1 * k
    C = C_base
    d = np.array([d_k])
    
    # Solve optimization (warm started from previous solution)
    problem = OptimizationProblem(A=A, b=b, C=C, d=d)
    solver = SNNSolver(problem, solver_config)
    
    t_start = time.time()
    result = solver.solve(x_current, verbose=False)
    t_solve = (time.time() - t_start) * 1000  # Convert to ms
    
    # Use solution as warm start for next iteration
    x_current = result.final_x
    
    # Store results
    solutions.append(x_current.copy())
    solve_times.append(t_solve)
    iterations.append(result.n_projections)
    
    # Print progress
    print(f"{k:<6} {-d_k:<10.3f} {result.final_objective:<12.6e} "
          f"{x_current[0]:<10.5f} {x_current[1]:<10.5f} "
          f"{t_solve:<10.3f} {result.n_projections:<8}")

print()
print("Statistics:")
print(f"  Average solve time: {np.mean(solve_times):.3f} ms")
print(f"  Max solve time: {np.max(solve_times):.3f} ms")
print(f"  Min solve time: {np.min(solve_times):.3f} ms")
print(f"  Average projections: {np.mean(iterations):.1f}")
print()

# Verify warm starting benefit
print("Warm starting analysis:")
print(f"  First solve (cold start): {solve_times[0]:.3f} ms, {iterations[0]} projections")
print(f"  Later solves benefit from warm start:")
for k in range(1, min(4, n_steps)):
    print(f"    Step {k}: {solve_times[k]:.3f} ms, {iterations[k]} projections")
print()

# Show solution trajectory
print("Solution trajectory:")
solutions_array = np.array(solutions)
for k in range(n_steps):
    print(f"  x[{k}] = [{solutions_array[k, 0]:.5f}, {solutions_array[k, 1]:.5f}]")


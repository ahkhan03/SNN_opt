"""
Example 1 (Advanced): Visual diagnostics for the 2D quadratic program.

This script solves a slight variant of the optimization problem in
`example1_simple_2d.py`, where the feasible region is shifted so that the
true unconstrained minimizer (0, 0) lies outside the feasible set. The solver
starts from a deliberately infeasible initial guess and visualizes:

1. Objective value vs. simulation time
2. Maximum constraint violation vs. simulation time
3. The 2D trajectory of the iterate relative to the feasible region

Run from the project root:
    python examples/example1_advanced_2d.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Ensure project root is on the path for local imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from snn_opt import OptimizationProblem, SNNSolver, SolverConfig  # noqa: E402


def build_problem():
    """Create the 2D QP with two linear constraints."""
    n = 2
    A = np.eye(n)
    b = np.zeros(n)
    C = np.array([[1.0, 2.0],
                  [-1.0, 3.0]])
    # Shift first constraint so that (0, 0) is infeasible: x1 + 2*x2 <= -0.2
    d = np.array([0.2, -1.0])
    return OptimizationProblem(A=A, b=b, C=C, d=d)


def solve_problem(problem: OptimizationProblem):
    """Solve the optimization problem from an infeasible starting point."""
    x0 = np.array([1.5, 1.5])  # Deliberately infeasible (violates both constraints)
    config = SolverConfig(t_end=200.0, k0=0.05, k1=0.1, max_step=0.1)
    solver = SNNSolver(problem, config)
    result = solver.solve(x0, verbose=False)
    print(result.summary())
    if result.spike_times.size:
        preview = min(5, result.spike_times.size)
        print(f"Spike preview (first {preview}):")
        for t, norm in zip(result.spike_times[:preview], result.spike_norms[:preview]):
            print(f"  t={t:.3f}, ||Δx||={norm:.3e}")
        print(f"Total projection distance: {result.total_projection_distance:.3e}\n")
    else:
        print("No spikes recorded.\n")
    return x0, result


def plot_progress(t: np.ndarray, objective: np.ndarray, violations: np.ndarray):
    """Create a two-panel plot for objective and constraint violation over time."""
    fig, (ax_obj, ax_constr) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax_obj.plot(t, objective, color="tab:blue", linewidth=2)
    ax_obj.set_ylabel("Objective value")
    ax_obj.set_title("Solver progress over time")
    ax_obj.grid(True, which="both", linestyle="--", alpha=0.3)

    ax_constr.plot(t, violations, color="tab:red", linewidth=2)
    ax_constr.set_xlabel("Time")
    ax_constr.set_ylabel("Max constraint violation")
    ax_constr.set_yscale("symlog", linthresh=1e-8)
    ax_constr.grid(True, which="both", linestyle="--", alpha=0.3)

    fig.tight_layout()
    return fig


def plot_trajectory(problem: OptimizationProblem, x0: np.ndarray, result):
    """Plot the search trajectory in 2D with constraint boundaries."""
    X = result.X

    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot constraint boundaries
    x1_vals = np.linspace(-0.2, 1.4, 200)
    line1 = (-0.2 - x1_vals) / 2.0
    line2 = (1.0 + x1_vals) / 3.0

    ax.plot(x1_vals, line1, label=r"$x_1 + 2x_2 = -0.2$", color="tab:green")
    ax.plot(x1_vals, line2, label=r"$-x_1 + 3x_2 = 1$", color="tab:purple")

    # Shade feasible region (below both lines)
    lower_bound = np.full_like(x1_vals, -0.5)
    upper_bound = np.minimum(line1, line2)
    ax.fill_between(x1_vals, lower_bound, upper_bound, color="lightgray", alpha=0.4, label="Feasible region")

    # Plot trajectory
    ax.plot(X[:, 0], X[:, 1], color="tab:blue", linewidth=2, label="Trajectory")
    ax.scatter([x0[0]], [x0[1]], color="tab:orange", s=60, label="Start", zorder=5)
    ax.scatter([result.final_x[0]], [result.final_x[1]], color="tab:red", s=60, label="Final", zorder=5)

    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_title("2D trajectory with constraint boundaries")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.5, 1.0)
    ax.legend(loc="upper right")

    fig.tight_layout()
    return fig


def main():
    problem = build_problem()
    x0, result = solve_problem(problem)

    plot_progress(result.t, result.objective_values, result.constraint_violations)
    plot_trajectory(problem, x0, result)

    plt.show()


if __name__ == "__main__":
    main()

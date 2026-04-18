"""
Example 6: Approximating equality constraints with inequality bands.

We solve a 2D quadratic program with the objective
    minimize 0.5 * ||x||^2 + b^T x (here b = 0)
subject to bands that emulate equality constraints x1 = a and x2 = b within
small slacks epsilon_x1 and epsilon_x2. Specifically we enforce

    x1 <= a + epsilon_x1
    x1 >= a - epsilon_x1  (implemented as -x1 <= -(a - epsilon_x1))
    x2 <= b + epsilon_x2
    x2 >= b - epsilon_x2

The unconstrained minimizer (0, 0) is intentionally chosen outside both
feasible bands so the solver must project onto the intersection of the two
constraint surfaces. The initial condition is also infeasible to highlight the
projection behaviour.

Run from the project root:
    python examples/example6_equality_constraint.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Ensure project root is on the path for local imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from snn_opt import OptimizationProblem, SNNSolver, SolverConfig  # noqa: E402


def main():
    # Equality targets and slacks
    a = 0.6
    epsilon_x1 = 0.02
    b_target = -0.35
    epsilon_x2 = 0.015

    # Quadratic cost minimize 0.5 * ||x||^2
    A = np.eye(2)
    b = np.zeros(2)

    # Inequality representation of the equality bands
    # Rows correspond to: x1 <= a + eps1, x1 >= a - eps1, x2 <= b + eps2, x2 >= b - eps2
    C = np.array([
        [1.0, 0.0],
        [-1.0, 0.0],
        [0.0, 1.0],
        [0.0, -1.0],
    ])
    d = np.array([
        -(a + epsilon_x1),   # x1 <= a + epsilon_x1
        a - epsilon_x1,      # x1 >= a - epsilon_x1
        -(b_target + epsilon_x2),  # x2 <= b + epsilon_x2
        b_target - epsilon_x2,     # x2 >= b - epsilon_x2
    ])

    problem = OptimizationProblem(A=A, b=b, C=C, d=d)

    # Deliberately infeasible initial condition (violates x1 upper bound and x2 <= 0.5)
    x0 = np.array([1.1, 0.0])

    config = SolverConfig(t_end=120.0, k0=0.01, k1=0.02, constraint_tol=1e-6)
    solver = SNNSolver(problem, config)
    result = solver.solve(x0, verbose=True)

    print()
    print("=" * 60)
    print("Example 6: Equality constraint approximation")
    print("=" * 60)
    print(f"Target equality: x1 ≈ {a} (±{epsilon_x1})")
    print(f"Target equality: x2 ≈ {b_target} (±{epsilon_x2})")
    print(f"Initial point: {x0}")
    print(result.summary())

    x1_low = a - epsilon_x1
    x1_high = a + epsilon_x1
    x2_low = b_target - epsilon_x2
    x2_high = b_target + epsilon_x2
    final_x1 = result.final_x[0]
    final_x2 = result.final_x[1]
    tol = config.constraint_tol
    within_x1 = abs(final_x1 - a) <= (epsilon_x1 + tol)
    within_x2 = abs(final_x2 - b_target) <= (epsilon_x2 + tol)
    print(f"x1 band: [{x1_low:.3f}, {x1_high:.3f}] -> final x1: {final_x1:.6f} "
        f"(within ±tol: {within_x1})")
    print(f"x2 band: [{x2_low:.3f}, {x2_high:.3f}] -> final x2: {final_x2:.6f} "
        f"(within ±tol: {within_x2})")
    max_violation = float(np.max(result.constraint_violations))
    final_violation = float(result.constraint_violations[-1])
    print(f"Max constraint violation along trajectory: {max_violation:.3e}")
    print(f"Final constraint violation: {final_violation:.3e}")
    if result.spike_times.size:
        preview = min(5, result.spike_times.size)
        print(f"Spike preview (first {preview}):")
        for t, norm in zip(result.spike_times[:preview], result.spike_norms[:preview]):
            print(f"  t={t:.3f}, ||Δx||={norm:.3e}")
        print(f"Total projection distance: {result.total_projection_distance:.3e}\n")
    else:
        print("No spikes recorded.\n")
    # Plot diagnostics
    _plot_progress(result.t, result.objective_values, result.constraint_violations)
    _plot_band_trajectory(a, epsilon_x1, b_target, epsilon_x2, x0, result)

    plt.show()


def _plot_progress(t: np.ndarray, objective: np.ndarray, violations: np.ndarray):
    """Plot objective value and constraint violation over time."""
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


def _plot_band_trajectory(a: float, epsilon_x1: float, b_target: float, epsilon_x2: float,
                          x0: np.ndarray, result):
    """Plot the iterate trajectory and equality bands in 2D."""
    X = result.X

    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot equality bands as shaded regions
    x1_band = [a - epsilon_x1, a + epsilon_x1]
    x2_band = [b_target - epsilon_x2, b_target + epsilon_x2]
    ax.axvspan(x1_band[0], x1_band[1], color="lightgray", alpha=0.4, label=r"$x_1$ band")
    ax.axhspan(x2_band[0], x2_band[1], color="mistyrose", alpha=0.4, label=r"$x_2$ band")

    # Plot trajectory
    ax.plot(X[:, 0], X[:, 1], color="tab:blue", linewidth=2, label="Trajectory")
    ax.scatter([x0[0]], [x0[1]], color="tab:orange", s=60, label="Start", zorder=5)
    ax.scatter([result.final_x[0]], [result.final_x[1]], color="tab:red", s=60, label="Final", zorder=5)

    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_title("Trajectory with equality band")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_xlim(a - 0.4, a + 0.6)
    ax.set_ylim(b_target - 0.6, b_target + 0.6)
    ax.legend(loc="upper right")

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    main()

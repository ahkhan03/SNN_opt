"""
Example 1 Basic: Raw SNN Solver with Inverted Constraints

Demonstrates the "raw" SNN search behavior on a 2D QP where the unconstrained
minimum (0,0) is OUTSIDE the feasible region, forcing the solution to lie
on the constraint boundary.

Problem:
    minimize ||x||²
    subject to:  x1 + 2*x2 >= 1   (origin is infeasible)
                -x1 + 3*x2 >= 1

The solver uses:
- Fixed step size k0 (no auto-tuning)
- Fixed projection step k1 (multiple spikes may be needed)
- No early stopping (runs all iterations)

This shows the raw "integrate-and-fire" dynamics.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
# Shared figure style, so this figure matches the benchmark suite and the site.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))

import numpy as np
import matplotlib.pyplot as plt

import figstyle
from snn_opt import OptimizationProblem, SNNSolver, SolverConfig, ConvergenceConfig

_ROOT = Path(__file__).resolve().parent.parent


def main():
    # Problem: minimize (1/2) x^T x
    # Subject to (INVERTED - origin is infeasible):
    #   x1 + 2*x2 >= 1   =>  -x1 - 2*x2 + 1 <= 0
    #  -x1 + 3*x2 >= 1   =>   x1 - 3*x2 + 1 <= 0
    
    n = 2
    A = np.eye(n)
    b = np.zeros(n)
    
    # Inverted constraints: C*x + d <= 0
    C = np.array([[-1.0, -2.0],   # -x1 - 2*x2 + 1 <= 0  (i.e., x1 + 2*x2 >= 1)
                  [1.0, -3.0]])   #  x1 - 3*x2 + 1 <= 0  (i.e., -x1 + 3*x2 >= 1)
    d = np.array([1.0, 1.0])
    
    problem = OptimizationProblem(A=A, b=b, C=C, d=d)
    
    # Initial guess (feasible - satisfies both constraints)
    x0 = np.array([2.0, 1.0])
    
    print("=" * 70)
    print("Example 1 Basic: Raw SNN Solver (Inverted Constraints)")
    print("=" * 70)
    print(f"Problem: minimize ||x||²")
    print(f"Subject to:  x1 + 2*x2 >= 1  (origin is OUTSIDE feasible region)")
    print(f"            -x1 + 3*x2 >= 1")
    print(f"Initial guess: {x0}")
    print()
    
    # Verify initial point is feasible
    g0 = C @ x0 + d
    print(f"Initial constraint values: {g0}  (should be <= 0)")
    print(f"Origin (0,0) constraint values: {C @ np.zeros(2) + d}  (positive = infeasible)")
    print()
    
    # RAW/BASIC MODE configuration
    raw_config = SolverConfig(
        k0=0.05,                          # Fixed step size
        k1=0.02,                          # Fixed projection step (small = more spikes)
        projection_method='fixed',        # Use fixed k1 (not adaptive)
        integration_method='euler',
        max_iterations=500,
        convergence=ConvergenceConfig(enable_early_stopping=False),  # Run all iterations
    )
    
    print("Raw mode settings:")
    print(f"  k0 = {raw_config.k0} (gradient step)")
    print(f"  k1 = {raw_config.k1} (projection step)")
    print(f"  max_iterations = {raw_config.max_iterations}")
    print()
    
    # Solve
    solver = SNNSolver(problem, raw_config)
    result = solver.solve(x0.copy(), verbose=False)
    
    print(result.summary())
    
    # Analytical solution: minimize ||x||² on the constraint boundaries
    # The optimum is at the intersection or on one constraint
    # Intersection of x1 + 2*x2 = 1 and -x1 + 3*x2 = 1:
    #   Adding: 5*x2 = 2  =>  x2 = 0.4
    #   x1 = 1 - 2*0.4 = 0.2
    x_analytical = np.array([0.2, 0.4])
    print(f"Analytical optimum: {x_analytical}")
    print(f"Analytical objective: {0.5 * np.dot(x_analytical, x_analytical):.6f}")
    print(f"Solver found: {result.final_x}")
    print(f"Error: {np.linalg.norm(result.final_x - x_analytical):.6e}")
    print()
    
    # ===== PLOTTING =====
    t = result.t
    X = result.X
    spikes = result.spike_times.astype(int)
    spikes = spikes[spikes < len(X)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.0))

    # ----- Panel (a): state vs iteration, with a spike rug -----
    ax1.plot(t, X[:, 0], color=figstyle.BLUE, linewidth=1.4, label="$x_1$")
    ax1.plot(t, X[:, 1], color=figstyle.VERMILION, linewidth=1.4, label="$x_2$")
    ax1.axhline(x_analytical[0], color=figstyle.BLUE, linestyle=":", linewidth=1.0,
                alpha=0.8, label=f"$x_1^\\star$ = {x_analytical[0]:.2f}")
    ax1.axhline(x_analytical[1], color=figstyle.VERMILION, linestyle=":", linewidth=1.0,
                alpha=0.8, label=f"$x_2^\\star$ = {x_analytical[1]:.2f}")

    # A rug of projection events along the bottom. Drawing one full-height
    # vertical line per spike (the previous version) washes the whole panel
    # orange once there are hundreds of them and hides the traces underneath.
    lo, hi = X.min() - 0.12, X.max() + 0.08
    ax1.eventplot(spikes, lineoffsets=lo + 0.03, linelengths=0.05,
                  colors=figstyle.PURPLE, linewidths=0.6, alpha=0.75)
    ax1.annotate(f"projection spikes ({len(spikes)})", xy=(0.02, lo + 0.07),
                 xycoords=("axes fraction", "data"), fontsize=8.0, color=figstyle.PURPLE)
    ax1.set_ylim(lo, hi)
    ax1.set_xlim(0, len(t))
    ax1.set_xlabel("iteration")
    ax1.set_ylabel("state value")
    figstyle.panel_title(ax1, "(a) state evolution")
    ax1.legend(loc="upper right", ncol=2)

    # ----- Panel (b): trajectory in state space -----
    x1_range = np.linspace(-0.4, 2.4, 300)
    x2_range = np.linspace(-0.4, 1.4, 300)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    Z = 0.5 * (X1**2 + X2**2)
    g1 = -X1 - 2 * X2 + 1
    g2 = X1 - 3 * X2 + 1
    feasible = (g1 <= 0) & (g2 <= 0)

    cs = ax2.contour(X1, X2, Z, levels=[0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0],
                     colors=figstyle.RULE, linewidths=0.7)
    ax2.clabel(cs, inline=True, fontsize=7, fmt="%.2f")
    ax2.contourf(X1, X2, feasible.astype(float), levels=[0.5, 1.5],
                 colors=[figstyle.GREEN], alpha=0.10)
    ax2.contour(X1, X2, g1, levels=[0], colors=[figstyle.MUTED], linewidths=1.0)
    ax2.contour(X1, X2, g2, levels=[0], colors=[figstyle.MUTED], linewidths=1.0,
                linestyles="--")

    ax2.plot(X[:, 0], X[:, 1], color=figstyle.INK, linewidth=0.9, alpha=0.85,
             zorder=3, label="trajectory")
    if spikes.size:
        ax2.scatter(X[spikes, 0], X[spikes, 1], s=14, color=figstyle.PURPLE,
                    alpha=0.75, zorder=4, label=f"spikes ({len(spikes)})")
    ax2.scatter(*X[0], s=44, color=figstyle.BLUE, marker="o", zorder=6, label="start")
    ax2.scatter(0, 0, s=44, color=figstyle.MUTED, marker="x", zorder=6,
                label="unconstrained min (infeasible)")
    ax2.scatter(*x_analytical, s=95, color=figstyle.VERMILION, marker="*", zorder=6,
                label=r"analytical $x^\star$")

    ax2.set_xlim(-0.3, 2.3)
    ax2.set_ylim(-0.3, 1.3)
    ax2.set_aspect("equal")
    ax2.set_xlabel("$x_1$")
    ax2.set_ylabel("$x_2$")
    figstyle.panel_title(ax2, "(b) trajectory over objective contours")
    ax2.legend(loc="upper right", fontsize=7.5)

    fig.tight_layout()
    output_path = Path(__file__).resolve().parent / "example1_basic_2d.png"
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    print(f"Figure saved to {output_path.relative_to(_ROOT)}")


if __name__ == "__main__":
    main()

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

import numpy as np
import matplotlib.pyplot as plt
from snn_opt import OptimizationProblem, SNNSolver, SolverConfig, ConvergenceConfig


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
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # ----- Plot 1: x1, x2 vs time -----
    ax1 = axes[0]
    t = result.t
    X = result.X
    
    ax1.plot(t, X[:, 0], 'b-', linewidth=1.5, label='$x_1$')
    ax1.plot(t, X[:, 1], 'r-', linewidth=1.5, label='$x_2$')
    
    # Mark spike events
    if len(result.spike_times) > 0:
        for spike_t in result.spike_times:
            ax1.axvline(x=spike_t, color='orange', alpha=0.3, linestyle='--', linewidth=0.8)
    
    # Mark analytical solution
    ax1.axhline(y=x_analytical[0], color='b', linestyle=':', alpha=0.5, label=f'$x_1^*$ = {x_analytical[0]:.2f}')
    ax1.axhline(y=x_analytical[1], color='r', linestyle=':', alpha=0.5, label=f'$x_2^*$ = {x_analytical[1]:.2f}')
    
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Variable Value', fontsize=12)
    ax1.set_title('State Evolution Over Time\n(orange lines = projection spikes)', fontsize=12)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, len(t)])
    
    # ----- Plot 2: x1 vs x2 with contours and constraints -----
    ax2 = axes[1]
    
    # Create grid for contours
    x1_range = np.linspace(-0.5, 2.5, 300)
    x2_range = np.linspace(-0.5, 1.5, 300)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    
    # Objective function contours: f(x) = 0.5 * (x1² + x2²)
    Z = 0.5 * (X1**2 + X2**2)
    
    # Constraints (in standard form Cx + d <= 0)
    # C1: -x1 - 2*x2 + 1 <= 0  =>  x1 + 2*x2 >= 1
    # C2:  x1 - 3*x2 + 1 <= 0  =>  -x1 + 3*x2 >= 1
    C1 = -X1 - 2*X2 + 1
    C2 = X1 - 3*X2 + 1
    feasible = (C1 <= 0) & (C2 <= 0)
    
    # Plot objective contours
    contour_levels = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0]
    cs = ax2.contour(X1, X2, Z, levels=contour_levels, colors='gray', alpha=0.6, linestyles='-')
    ax2.clabel(cs, inline=True, fontsize=8, fmt='%.2f')
    
    # Shade feasible region
    ax2.contourf(X1, X2, feasible.astype(float), levels=[0.5, 1.5], 
                 colors=['lightgreen'], alpha=0.4)
    
    # Plot constraint boundaries (using purple and teal to avoid confusion with x1/x2 colors)
    ax2.contour(X1, X2, C1, levels=[0], colors=['purple'], linewidths=2, linestyles=['-'])
    ax2.contour(X1, X2, C2, levels=[0], colors=['teal'], linewidths=2, linestyles=['-'])
    
    # Plot trajectory
    ax2.plot(X[:, 0], X[:, 1], 'k-', alpha=0.6, linewidth=1, label='Trajectory')
    
    # Mark key points
    ax2.plot(X[0, 0], X[0, 1], 'go', markersize=12, label='Start', zorder=10)
    ax2.plot(X[-1, 0], X[-1, 1], 'r*', markersize=15, label='End', zorder=10)
    ax2.plot(0, 0, 'kx', markersize=12, markeredgewidth=2, label='Unconstrained min (infeasible)')
    ax2.plot(x_analytical[0], x_analytical[1], 'b^', markersize=10, 
             label=f'Analytical opt ({x_analytical[0]:.2f}, {x_analytical[1]:.2f})', zorder=10)
    
    # Mark spike locations
    if len(result.spike_times) > 0:
        spike_indices = result.spike_times.astype(int)
        spike_indices = spike_indices[spike_indices < len(X)]
        if len(spike_indices) > 0:
            ax2.scatter(X[spike_indices, 0], X[spike_indices, 1], 
                       c='orange', s=30, alpha=0.8, zorder=5, 
                       edgecolors='black', linewidths=0.5, label='Spikes')
    
    ax2.set_xlabel('$x_1$', fontsize=12)
    ax2.set_ylabel('$x_2$', fontsize=12)
    ax2.set_title('Trajectory in State Space\n(contours = objective, green = feasible region)', fontsize=12)
    ax2.legend(loc='lower right', fontsize=9)
    ax2.set_xlim([-0.3, 2.3])
    ax2.set_ylim([-0.3, 1.3])
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(__file__).parent / 'example1_basic_2d.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved to {output_path}")
    plt.show()


if __name__ == "__main__":
    main()

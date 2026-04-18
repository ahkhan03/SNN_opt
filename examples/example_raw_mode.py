"""
Example: Raw Mode vs Optimized Mode Comparison

This demonstrates the SNN solver's "raw" search behavior without optimizations,
compared to the default optimized configuration.

Raw Mode disables:
- Adaptive projection (uses fixed k1 step instead of exact boundary projection)
- Auto step-size computation (uses manual k0)
- Early stopping (runs all iterations)

This shows more of the neuromorphic "integrate-and-fire" dynamics where
the solver may overshoot and require multiple spikes to reach the boundary.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import matplotlib.pyplot as plt
from snn_opt import OptimizationProblem, SNNSolver, SolverConfig, ConvergenceConfig


def create_raw_config(k0: float = 0.1, k1: float = 0.05, max_iterations: int = 500) -> SolverConfig:
    """
    Create a 'raw' configuration that shows unoptimized SNN behavior.
    
    This disables:
    - Adaptive projection (uses fixed k1 steps)
    - Auto k0 computation (uses provided k0)
    - Early stopping (runs all iterations)
    
    Parameters
    ----------
    k0 : float
        Fixed gradient descent step size
    k1 : float
        Fixed projection step size (may need multiple iterations to reach boundary)
    max_iterations : int
        Number of iterations to run
    
    Returns
    -------
    SolverConfig
        Configuration for raw mode
    """
    conv_config = ConvergenceConfig(
        enable_early_stopping=False,  # Run all iterations
    )
    
    return SolverConfig(
        k0=k0,                           # Fixed step size (no auto-compute)
        k1=k1,                           # Fixed projection step
        projection_method='fixed',       # Use fixed k1 instead of adaptive
        integration_method='euler',
        max_iterations=max_iterations,
        convergence=conv_config,
    )


def create_optimized_config(max_iterations: int = 500) -> SolverConfig:
    """
    Create the default optimized configuration.
    
    This enables:
    - Adaptive projection (exact step to boundary)
    - Auto k0 from Lipschitz constant
    - Early stopping with multiple convergence criteria
    """
    conv_config = ConvergenceConfig(
        enable_early_stopping=True,
        check_every=50,
        min_iterations=100,
        patience=3,
    )
    
    return SolverConfig(
        k0=None,                         # Auto-compute from Lipschitz constant
        projection_method='adaptive',    # Exact projection to boundary
        integration_method='euler',
        max_iterations=max_iterations,
        k0_scale=0.5,
        convergence=conv_config,
    )


def main():
    # Problem: minimize (1/2) x^T x subject to:
    #   x1 + 2*x2 <= 1
    #  -x1 + 3*x2 <= 1
    n = 2
    A = np.eye(n)
    b = np.zeros(n)
    C = np.array([[1.0, 2.0],
                  [-1.0, 3.0]])
    d = np.array([-1.0, -1.0])
    
    problem = OptimizationProblem(A=A, b=b, C=C, d=d)
    
    # Initial guess (infeasible)
    x0 = np.array([1.0, 1.0])
    
    print("=" * 70)
    print("SNN Solver: Raw Mode vs Optimized Mode Comparison")
    print("=" * 70)
    print(f"Problem: minimize ||x||²")
    print(f"Subject to: x1 + 2*x2 <= 1")
    print(f"           -x1 + 3*x2 <= 1")
    print(f"Initial guess: {x0}")
    print()
    
    # ===== RAW MODE =====
    print("-" * 35)
    print("RAW MODE")
    print("-" * 35)
    print("Settings: k0=0.1, k1=0.05 (fixed), no early stopping")
    print()
    
    raw_config = create_raw_config(k0=0.1, k1=0.05, max_iterations=300)
    raw_solver = SNNSolver(problem, raw_config)
    raw_result = raw_solver.solve(x0.copy(), verbose=False)
    
    print(raw_result.summary())
    
    # ===== OPTIMIZED MODE =====
    print("-" * 35)
    print("OPTIMIZED MODE")
    print("-" * 35)
    print("Settings: k0=auto, adaptive projection, early stopping")
    print()
    
    opt_config = create_optimized_config(max_iterations=300)
    opt_solver = SNNSolver(problem, opt_config)
    opt_result = opt_solver.solve(x0.copy(), verbose=False)
    
    print(opt_result.summary())
    
    # ===== COMPARISON =====
    print("=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"{'Metric':<30} {'Raw':>15} {'Optimized':>15}")
    print("-" * 70)
    print(f"{'Iterations used':<30} {raw_result.iterations_used:>15} {opt_result.iterations_used:>15}")
    print(f"{'Total projections (spikes)':<30} {raw_result.n_projections:>15} {opt_result.n_projections:>15}")
    print(f"{'Total projection distance':<30} {raw_result.total_projection_distance:>15.6f} {opt_result.total_projection_distance:>15.6f}")
    print(f"{'Final objective':<30} {raw_result.final_objective:>15.6e} {opt_result.final_objective:>15.6e}")
    print(f"{'Final proj. gradient norm':<30} {raw_result.final_proj_grad_norm:>15.6e} {opt_result.final_proj_grad_norm:>15.6e}")
    print(f"{'Converged':<30} {str(raw_result.converged):>15} {str(opt_result.converged):>15}")
    print()
    
    # ===== VISUALIZE =====
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot feasible region
        x1_range = np.linspace(-0.5, 1.5, 200)
        x2_range = np.linspace(-0.5, 1.0, 200)
        X1, X2 = np.meshgrid(x1_range, x2_range)
        
        # Constraints
        C1 = X1 + 2*X2 - 1  # <= 0
        C2 = -X1 + 3*X2 - 1  # <= 0
        feasible = (C1 <= 0) & (C2 <= 0)
        
        for ax, result, title in [(axes[0], raw_result, "Raw Mode Trajectory"),
                                   (axes[1], opt_result, "Optimized Mode Trajectory")]:
            # Feasible region
            ax.contourf(X1, X2, feasible.astype(float), levels=[0.5, 1.5], 
                        colors=['lightgreen'], alpha=0.3)
            ax.contour(X1, X2, C1, levels=[0], colors=['blue'], linestyles=['--'])
            ax.contour(X1, X2, C2, levels=[0], colors=['red'], linestyles=['--'])
            
            # Trajectory
            X = result.X
            ax.plot(X[:, 0], X[:, 1], 'k-', alpha=0.5, linewidth=0.5, label='Trajectory')
            ax.plot(X[0, 0], X[0, 1], 'go', markersize=10, label='Start')
            ax.plot(X[-1, 0], X[-1, 1], 'r*', markersize=15, label='End')
            
            # Mark spikes
            if len(result.spike_times) > 0:
                spike_indices = result.spike_times.astype(int)
                spike_indices = spike_indices[spike_indices < len(X)]
                if len(spike_indices) > 0:
                    ax.scatter(X[spike_indices, 0], X[spike_indices, 1], 
                              c='orange', s=20, alpha=0.7, zorder=5, label='Spikes')
            
            ax.set_xlabel('x1')
            ax.set_ylabel('x2')
            ax.set_title(f"{title}\n(iters={result.iterations_used}, spikes={result.n_projections})")
            ax.legend(loc='upper right', fontsize=8)
            ax.set_xlim(-0.3, 1.3)
            ax.set_ylim(-0.3, 0.8)
            ax.set_aspect('equal')
        
        # Convergence plot
        ax = axes[2]
        ax.semilogy(raw_result.objective_values, 'b-', alpha=0.7, label='Raw Mode')
        ax.semilogy(opt_result.objective_values, 'r-', alpha=0.7, label='Optimized Mode')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Objective Value (log scale)')
        ax.set_title('Convergence Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(Path(__file__).parent / 'raw_vs_optimized.png', dpi=150)
        print("Plot saved to examples/raw_vs_optimized.png")
        plt.show()
        
    except Exception as e:
        print(f"Plotting skipped: {e}")


if __name__ == "__main__":
    main()

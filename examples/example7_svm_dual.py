"""
Example 7: SVM Dual Problem

Demonstrates the SNN solver on a Support Vector Machine dual optimization problem.
This showcases:
- Auto k0 computation from Hessian eigenvalue
- Box constraint clipping (0 <= alpha <= C)
- Equality constraint handling (y^T alpha = 0)

The SVM dual problem:
    minimize    (1/2) alpha^T Q alpha - 1^T alpha
    subject to  y^T alpha = 0
                0 <= alpha_i <= C

where Q_ij = y_i y_j K(x_i, x_j) is the label-weighted kernel matrix.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from snn_opt import SNNSolver, SolverConfig, OptimizationProblem


def rbf_kernel(X, gamma=1.0):
    """Compute RBF kernel matrix."""
    sq_dists = np.sum(X**2, axis=1, keepdims=True) + np.sum(X**2, axis=1) - 2 * X @ X.T
    return np.exp(-gamma * sq_dists)


def main():
    print("=" * 60)
    print("Example 7: SVM Dual Problem")
    print("=" * 60)
    
    # Generate synthetic data (two-class classification)
    np.random.seed(42)
    n_per_class = 25
    
    # Class +1: centered at (1, 1)
    X_pos = np.random.randn(n_per_class, 2) * 0.5 + np.array([1, 1])
    # Class -1: centered at (-1, -1)
    X_neg = np.random.randn(n_per_class, 2) * 0.5 + np.array([-1, -1])
    
    X = np.vstack([X_pos, X_neg])
    y = np.array([1] * n_per_class + [-1] * n_per_class, dtype=float)
    
    n = len(y)
    C = 1.0  # SVM regularization parameter
    
    print(f"\nDataset: {n} samples, 2 classes")
    print(f"SVM C parameter: {C}")
    
    # Compute kernel matrix
    gamma = 0.5
    K = rbf_kernel(X, gamma=gamma)
    print(f"RBF kernel gamma: {gamma}")
    
    # Q matrix: Q_ij = y_i y_j K_ij
    Q = np.outer(y, y) * K
    
    # Add small regularization for numerical stability
    Q += 1e-8 * np.eye(n)
    
    # SVM dual: minimize (1/2) alpha^T Q alpha - 1^T alpha
    b = -np.ones(n)
    
    # Only equality constraint as linear inequality (box handled by clipping)
    # y^T alpha = 0 converted to: y^T alpha <= 0 AND -y^T alpha <= 0
    C_eq = np.vstack([
        y.reshape(1, -1),
        -y.reshape(1, -1),
    ])
    d_eq = np.array([0.0, 0.0])
    
    print(f"\nProblem dimensions:")
    print(f"  Variables: {n}")
    print(f"  Equality constraints: 1 (y^T alpha = 0)")
    print(f"  Box constraints: 0 <= alpha <= {C} (handled by clipping)")
    
    # Create problem
    problem = OptimizationProblem(A=Q, b=b, C=C_eq, d=d_eq)
    
    # Configure solver with auto k0 and box clipping
    config = SolverConfig(
        k0=None,           # Auto-compute from Hessian eigenvalue
        k0_scale=0.5,      # Conservative scaling
        max_iterations=2000,
        integration_method='euler',
        projection_method='adaptive',
        lower_bound=0.0,   # alpha >= 0
        upper_bound=C,     # alpha <= C
    )
    
    # Solve
    solver = SNNSolver(problem, config)
    print(f"\nAuto-computed k0: {solver._k0:.6e}")
    
    alpha0 = np.zeros(n)
    result = solver.solve(alpha0, verbose=False)
    
    # Results
    alpha = result.final_x
    
    print(f"\n--- Results ---")
    print(f"Final objective: {result.final_objective:.4f}")
    print(f"Converged: {result.converged}")
    print(f"Convergence reason: {result.convergence_reason}")
    print(f"Iterations used: {result.iterations_used} / {config.max_iterations}")
    print(f"Projected gradient norm: {result.final_proj_grad_norm:.2e}")
    print(f"Total projections: {result.n_projections}")
    
    # Check constraints
    eq_violation = abs(y @ alpha)
    box_violations = np.sum(alpha < -1e-6) + np.sum(alpha > C + 1e-6)
    print(f"\nConstraint satisfaction:")
    print(f"  |y^T alpha| = {eq_violation:.6f}")
    print(f"  Box violations: {box_violations}")
    
    # Support vectors
    sv_mask = alpha > 1e-5
    free_sv_mask = (alpha > 1e-5) & (alpha < C - 1e-5)
    
    print(f"\nSupport vectors:")
    print(f"  Total: {np.sum(sv_mask)}")
    print(f"  Free (0 < alpha < C): {np.sum(free_sv_mask)}")
    print(f"  Bounded (alpha = C): {np.sum(alpha > C - 1e-5)}")
    
    # Compute decision function and accuracy
    free_sv_idx = np.where(free_sv_mask)[0]
    if len(free_sv_idx) > 0:
        # Compute bias from free support vectors
        K_sv = K[free_sv_idx, :]
        bias_vals = y[free_sv_idx] - (alpha * y) @ K_sv.T
        bias = np.mean(bias_vals)
        
        # Decision function
        decision = (alpha * y) @ K + bias
        y_pred = np.sign(decision)
        
        accuracy = np.mean(y_pred == y) * 100
        print(f"\nTraining accuracy: {accuracy:.1f}%")
    
    print("\n" + "=" * 60)
    print("SVM dual solved successfully with auto k0 + box clipping!")
    print("=" * 60)


if __name__ == '__main__':
    main()


"""snn_opt — A spiking neural network solver for constrained convex optimization.

Solves problems of the form

    minimize    (1/2) x^T A x + b^T x
    subject to  C x + d <= 0

by alternating gradient descent with discrete boundary projections, the
optimization equivalent of LIF integrate-and-fire dynamics. The framework
is described in:

    Mancoo, Boerlin & Machens, *Understanding spiking networks through
    convex optimization*, NeurIPS 2020.

Public API
----------
- ``OptimizationProblem`` — problem definition (A, b, C, d)
- ``SolverConfig`` — solver hyperparameters (with sensible auto-defaults)
- ``ConvergenceConfig`` — early-stopping criteria
- ``SolverResult`` — solution + diagnostics (trajectory, spike events, …)
- ``SNNSolver`` — full solver class (use for repeated/warm-started solves)
- ``solve_qp`` — convenience function for one-shot QPs

See ``docs/applications.md`` for published work that uses this solver.
"""

from .solver import (
    ConvergenceConfig,
    OptimizationProblem,
    SNNSolver,
    SolverConfig,
    SolverResult,
    solve_qp,
)

__version__ = "0.1.0"

__all__ = [
    "ConvergenceConfig",
    "OptimizationProblem",
    "SNNSolver",
    "SolverConfig",
    "SolverResult",
    "solve_qp",
    "__version__",
]

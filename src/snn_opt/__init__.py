"""snn_opt — A spiking neural network solver for constrained convex optimization.

Solves problems of the form

    minimize    (1/2) x^T A x + b^T x
    subject to  C x + d <= 0

by alternating gradient descent with discrete boundary projections, the
optimization equivalent of LIF integrate-and-fire dynamics. The framework
is described in:

    Mancoo, Keemink & Machens, *Understanding spiking networks through
    convex optimization*, NeurIPS 2020.

Public API
----------
- ``OptimizationProblem`` — problem definition (A, b, C, d)
- ``SolverConfig`` — solver hyperparameters (with sensible auto-defaults)
- ``ConvergenceConfig`` — early-stopping criteria
- ``SolverResult`` — solution + diagnostics (trajectory, spike events, …)
- ``SNNSolver`` — full solver class (use for repeated/warm-started solves)
- ``solve_qp`` — convenience function for one-shot QPs (set ``backend='c'`` for
  the compiled kernel; see ``SolverConfig.backend`` for the 'c'/'c_serial'/
  'c_openmp' variants)
- ``Transform`` / ``EigenbasisTransform`` — problem transforms (the transform
  axis); opt in via ``SolverConfig.transform`` (e.g. ``transform='eigenbasis'``)

See ``docs/applications.md`` for published work that uses this solver.
"""

from .nonlinear import (
    AffineSubspaceProjector,
    CutterCandidate,
    DykstraProjector,
    LiftedSOCResult,
    ProjectorCandidate,
    affine_cutter,
    ball_projector,
    dykstra_intersection_projector,
    dykstra_projector,
    halfspace_projector,
    joint_dykstra_projector,
    joint_projector,
    lift_soc_l1,
    lift_soc_l2,
    psd_cone_projector,
    psd_projector,
    scaled_soc_projector,
    soc_projector,
    spectral_ball_cutter,
    spectral_ball_projector,
    spectral_norm_cutter,
)
from .solver import (
    ConvergenceConfig,
    OptimizationProblem,
    SNNSolver,
    SolverConfig,
    SolverResult,
    solve_qp,
)
from .transforms import EigenbasisTransform, Transform

# Keep in sync with the version in pyproject.toml and CITATION.cff.
__version__ = "0.8.0"

__all__ = [
    "ConvergenceConfig",
    "OptimizationProblem",
    "SNNSolver",
    "SolverConfig",
    "SolverResult",
    "solve_qp",
    "CutterCandidate",
    "ProjectorCandidate",
    "AffineSubspaceProjector",
    "DykstraProjector",
    "affine_cutter",
    "halfspace_projector",
    "ball_projector",
    "soc_projector",
    "scaled_soc_projector",
    "psd_cone_projector",
    "psd_projector",
    "dykstra_projector",
    "joint_dykstra_projector",
    "joint_projector",
    "dykstra_intersection_projector",
    "spectral_norm_cutter",
    "spectral_ball_cutter",
    "spectral_ball_projector",
    "LiftedSOCResult",
    "lift_soc_l1",
    "lift_soc_l2",
    "Transform",
    "EigenbasisTransform",
    "__version__",
]

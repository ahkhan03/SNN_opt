"""Joint feasibility checks for the extended winner-take-all sweep."""

import numpy as np

from snn_opt import (
    ConvergenceConfig,
    CutterCandidate,
    OptimizationProblem,
    SNNSolver,
    SolverConfig,
    ball_projector,
    soc_projector,
)


def test_every_post_sweep_iterate_is_feasible():
    # The rows, scalar bounds, a nonlinear cutter, a subset ball, and an SOC
    # all compete in one family.  Every stored state is after a complete sweep.
    n = 6
    A = np.eye(n)
    b = -np.ones(n)
    C = np.array([[1.0, 1.0, 0, 0, 0, 0]], dtype=float)
    d = np.array([-0.5])
    cutter = CutterCandidate(
        value=lambda x: float(0.5 * (x[0] ** 2 + x[1] ** 2 - 0.25)),
        jacobian=lambda x: np.array([x[0], x[1], 0, 0, 0, 0], dtype=float),
        name="disk-cutter",
    )
    candidates = (cutter, ball_projector([2, 3], 0.5), soc_projector(5, [4]))
    problem = OptimizationProblem(A, b, C, d,
                                  nonlinear_candidates=candidates)
    cfg = SolverConfig(
        k0=0.05,
        lower_bound=0.0,
        upper_bound=1.0,
        max_iterations=40,
        constraint_tol=1e-8,
        convergence=ConvergenceConfig(enable_early_stopping=False,
                                       feasibility_tol=1e-7),
    )
    result = SNNSolver(problem, cfg).solve(np.zeros(n))
    assert not result.projection_budget_exhausted
    assert result.max_distance_rows <= 1e-7
    assert result.max_violation_box <= 1e-7
    for x in result.X[1:]:
        assert problem.max_violation(x) / max(1.0, np.linalg.norm(C[0])) <= 1e-7
        assert np.max(np.maximum(0.0 - x, 0.0)) <= 1e-7
        assert np.max(np.maximum(x - 1.0, 0.0)) <= 1e-7
        disk_value = 0.5 * (x[0] ** 2 + x[1] ** 2 - 0.25)
        disk_grad_norm = np.linalg.norm(x[:2])
        assert max(disk_value, 0.0) / max(disk_grad_norm, 1e-15) <= 1e-7
        assert np.linalg.norm(x[[2, 3]]) <= 0.5 + 1e-7
        assert abs(x[4]) <= x[5] + 1e-7
    assert result.joint_feasible


def test_projection_budget_continuation_is_extended_opt_in_only():
    """A capped nonlinear sweep may continue, while released rows still abort."""
    # Two independent cutters need two events to repair this state.  A budget
    # of one therefore leaves a positive violation after the first sweep.
    cutters = tuple(
        CutterCandidate(
            value=lambda x, i=i: float(-x[i]),
            jacobian=lambda x, i=i: -np.eye(2)[i],
            name=f"nonnegative-{i}",
        )
        for i in range(2)
    )
    A = np.zeros((2, 2))
    b = np.zeros(2)
    x0 = np.array([-1.0, -1.0])
    common = dict(
        k0=0.1,
        max_iterations=4,
        max_projection_iters=1,
        constraint_tol=1e-10,
        convergence=ConvergenceConfig(enable_early_stopping=False,
                                       feasibility_tol=1e-10),
    )

    # The released polyhedral branch does not consult the continuation switch.
    released = SNNSolver(
        OptimizationProblem(A, b, -np.eye(2), np.zeros(2)),
        SolverConfig(**common, continue_after_projection_budget=True),
    ).solve(x0)
    assert released.projection_budget_exhausted
    assert released.convergence_reason == "projection_budget_exhausted"

    extended_problem = OptimizationProblem(
        A, b, np.zeros((0, 2)), np.zeros(0), nonlinear_candidates=cutters)
    extended = SNNSolver(
        extended_problem,
        SolverConfig(**common, continue_after_projection_budget=True),
    ).solve(x0)
    assert not extended.projection_budget_exhausted
    assert extended.convergence_reason == "max_iterations"
    assert extended.projection_truncated_sweeps >= 1
    assert extended.final_x[0] >= -1e-10 and extended.final_x[1] >= -1e-10

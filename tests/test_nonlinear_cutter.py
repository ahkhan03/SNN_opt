"""Regression tests for differentiable-inequality candidate cutters."""

import numpy as np

from snn_opt import (
    ConvergenceConfig,
    CutterCandidate,
    OptimizationProblem,
    SNNSolver,
    SolverConfig,
    affine_cutter,
)


def _run(problem, *, observe=True):
    cfg = SolverConfig(
        k0=0.25,
        max_iterations=12,
        constraint_tol=1e-12,
        observe_projection_events=observe,
        convergence=ConvergenceConfig(enable_early_stopping=False),
    )
    return SNNSolver(problem, cfg).solve(np.array([1.0, 1.0]))


def test_affine_cutter_matches_row_path_bitwise():
    """A tied affine cutter must not perturb the released row winner."""
    A = np.eye(2)
    b = np.array([-2.0, -1.0])
    C = np.array([[1.0, 0.0], [0.0, 1.0]])
    d = np.zeros(2)
    base = OptimizationProblem(A, b, C, d)
    # Keep the rows in the family as the frozen first candidates; the affine
    # adapters are exact ties and therefore cannot displace them.
    candidates = tuple(affine_cutter(C[i], d[i], name=f"row-cutter-{i}")
                       for i in range(2))
    extended = OptimizationProblem(A, b, C, d,
                                   nonlinear_candidates=candidates)
    plain = _run(base)
    ext = _run(extended)
    assert np.array_equal(plain.X, ext.X)
    assert plain.spike_constraints == ext.spike_constraints
    assert plain.projection_event_digest == ext.projection_event_digest
    assert ext.nonlinear_event_counts == {}


def test_positive_scaling_preserves_trajectory_and_events():
    """Positive rescaling of cutter values and Jacobians is geometric only."""
    A = np.diag([2.0, 4.0])
    b = np.array([2.0, 4.0])
    factors = np.array([1.0, 1.0])

    def make(scale):
        c0 = CutterCandidate(
            lambda x, s=scale[0]: float(s * (-x[0])),
            lambda x, s=scale[0]: np.array([-s, 0.0]),
            name="x",
        )
        c1 = CutterCandidate(
            lambda x, s=scale[1]: float(s * (-x[1])),
            lambda x, s=scale[1]: np.array([0.0, -s]),
            name="y",
        )
        return OptimizationProblem(
            A, b, np.zeros((0, 2)), np.zeros(0),
            nonlinear_candidates=(c0, c1),
        )

    r1 = _run(make(factors))
    r2 = _run(make(np.array([1e-3, 1e3])))
    np.testing.assert_allclose(r1.X, r2.X, rtol=0.0, atol=2e-15)
    assert r1.spike_event_kinds == r2.spike_event_kinds
    np.testing.assert_array_equal(r1.spike_event_indices, r2.spike_event_indices)
    assert r1.nonlinear_event_counts["cutter"] == r2.nonlinear_event_counts["cutter"]

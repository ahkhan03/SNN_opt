"""Frozen-path regression for an explicitly empty candidate tuple."""

import numpy as np

from snn_opt import ConvergenceConfig, OptimizationProblem, SNNSolver, SolverConfig


def test_empty_candidates_preserve_frozen_polyhedral_path():
    A = np.diag([1.0, 2.0])
    b = np.array([-2.0, 1.0])
    C = np.array([[1.0, 1.0], [-1.0, 0.0]])
    d = np.array([-0.5, 0.0])
    cfg = SolverConfig(
        k0=0.125,
        lower_bound=0.0,
        upper_bound=1.0,
        max_iterations=24,
        constraint_tol=1e-12,
        observe_projection_events=True,
        convergence=ConvergenceConfig(enable_early_stopping=False),
    )
    p_implicit = OptimizationProblem(A, b, C, d)
    p_explicit = OptimizationProblem(A, b, C, d, nonlinear_candidates=())
    r1 = SNNSolver(p_implicit, cfg).solve(np.array([0.75, 0.25]))
    r2 = SNNSolver(p_explicit, cfg).solve(np.array([0.75, 0.25]))
    assert np.array_equal(r1.X, r2.X)
    assert r1.n_projections == r2.n_projections
    assert r1.projection_event_digest == r2.projection_event_digest
    assert r1.projection_first_candidate_id == r2.projection_first_candidate_id
    assert r1.projection_last_candidate_id == r2.projection_last_candidate_id
    np.testing.assert_array_equal(r1.explicit_row_event_counts,
                                  r2.explicit_row_event_counts)
    np.testing.assert_array_equal(r1.implicit_lower_event_counts,
                                  r2.implicit_lower_event_counts)
    np.testing.assert_array_equal(r1.implicit_upper_event_counts,
                                  r2.implicit_upper_event_counts)
    assert r1.spike_event_kinds == r2.spike_event_kinds
    np.testing.assert_array_equal(r1.spike_event_indices,
                                  r2.spike_event_indices)

"""Public API and opt-in dispatch guards."""

import numpy as np
import pytest

from snn_opt import (
    ConvergenceConfig,
    CutterCandidate,
    OptimizationProblem,
    ProjectorCandidate,
    SNNSolver,
    SolverConfig,
    ball_projector,
)


def _problem():
    candidate = ball_projector([0], 1.0)
    return OptimizationProblem(
        np.eye(1), np.zeros(1), np.zeros((0, 1)), np.zeros(0),
        nonlinear_candidates=(candidate,),
    )


@pytest.mark.parametrize(
    "overrides, phrase",
    [
        ({"record_trajectory": False}, "record_trajectory"),
        ({"transform": "eigenbasis"}, "transform"),
        ({"integration_method": "ivp"}, "integration_method"),
        ({"projection_method": "fixed"}, "projection_method"),
    ],
)
def test_unsupported_combinations_raise(overrides, phrase):
    with pytest.raises(ValueError, match=phrase):
        SNNSolver(_problem(), SolverConfig(
            convergence=ConvergenceConfig(enable_early_stopping=False),
            **overrides,
        ))


def test_builtin_projector_is_accepted_by_native_constructor():
    solver = SNNSolver(_problem(), SolverConfig(
        backend="c",
        convergence=ConvergenceConfig(enable_early_stopping=False),
    ))
    assert solver._native_descriptor_cache is not None
    assert solver._native_descriptor_cache[0].shape == (1, 11)


def test_callback_projector_is_rejected_by_native_constructor():
    callback = CutterCandidate(
        value=lambda x: float(x[0] - 1.0),
        jacobian=lambda x: np.array([1.0]),
    )
    problem = OptimizationProblem(
        np.eye(1), np.zeros(1), np.zeros((0, 1)), np.zeros(0),
        nonlinear_candidates=(callback,),
    )
    with pytest.raises(ValueError, match=r"candidate 0.*backend='python'"):
        SNNSolver(problem, SolverConfig(
            backend="c",
            convergence=ConvergenceConfig(enable_early_stopping=False),
        ))


def test_unknown_projector_certificate_is_reported_not_available():
    unknown = ProjectorCandidate(lambda x: np.asarray(x, dtype=float), name="opaque")
    problem = OptimizationProblem(
        np.eye(1), np.zeros(1), np.zeros((0, 1)), np.zeros(0),
        nonlinear_candidates=(unknown,),
    )
    result = SNNSolver(problem, SolverConfig(
        max_iterations=2,
        convergence=ConvergenceConfig(enable_early_stopping=False),
    )).solve(np.zeros(1))
    assert result.kkt_fit_status == "not_available"

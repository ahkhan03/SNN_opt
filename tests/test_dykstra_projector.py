"""Focused checks for the cold Dykstra intersection projector."""

import numpy as np
import pytest

from snn_opt import (
    AffineSubspaceProjector,
    ConvergenceConfig,
    DykstraProjector,
    OptimizationProblem,
    SNNSolver,
    SolverConfig,
    ball_projector,
    box_projector,
    halfspace_projector,
    joint_dykstra_projector,
    psd_cone_projector,
    soc_projector,
)


def test_psd_svec_clip_matches_eigh_for_indefinite_and_rank_one_inputs():
    projector = psd_cone_projector(3)
    rng = np.random.default_rng(4)
    for matrix in (
        np.array([[1.0, 2.0, -1.0], [2.0, -3.0, 0.5], [-1.0, 0.5, 0.2]]),
        np.outer(rng.normal(size=3), rng.normal(size=3)),
    ):
        matrix = 0.5 * (matrix + matrix.T)
        eigenvalues, vectors = np.linalg.eigh(matrix)
        expected = (vectors * np.clip(eigenvalues, 0.0, None)) @ vectors.T
        # svec ordering is (00, 01, 02, 11, 12, 22).
        packed = np.array([
            matrix[0, 0], np.sqrt(2.0) * matrix[0, 1],
            np.sqrt(2.0) * matrix[0, 2], matrix[1, 1],
            np.sqrt(2.0) * matrix[1, 2], matrix[2, 2],
        ])
        projected = projector.project(packed)
        unpacked = np.array([
            [projected[0], projected[1] / np.sqrt(2.0), projected[2] / np.sqrt(2.0)],
            [projected[1] / np.sqrt(2.0), projected[3], projected[4] / np.sqrt(2.0)],
            [projected[2] / np.sqrt(2.0), projected[4] / np.sqrt(2.0), projected[5]],
        ])
        np.testing.assert_allclose(unpacked, expected, atol=2e-13)


def test_dykstra_row_soc_projection_matches_clarabel():
    cp = pytest.importorskip("cvxpy")
    x0 = np.array([-0.3, 2.0, -1.0])
    candidate = joint_dykstra_projector(
        np.array([[-1.0, 0.0, 0.0]]), np.array([0.5]),
        cones=(soc_projector(0, [1, 2]),),
    )
    projected = candidate.project(x0)
    x = cp.Variable(3)
    problem = cp.Problem(
        cp.Minimize(0.5 * cp.sum_squares(x - x0)),
        [-x[0] + 0.5 <= 0, cp.norm(x[1:]) <= x[0]],
    )
    problem.solve(solver=cp.CLARABEL, tol_gap_abs=1e-11,
                  tol_gap_rel=1e-11, tol_feas=1e-11)
    np.testing.assert_allclose(projected, x.value, atol=2e-8)
    assert candidate.last_diagnostics["converged"]
    assert candidate.last_diagnostics["projection_events"] >= 2


def test_dykstra_affine_psd_projection_matches_clarabel():
    cp = pytest.importorskip("cvxpy")
    root2 = np.sqrt(2.0)
    # Packed symmetric 2x2 state [X00, sqrt(2) X01, X11].
    point = np.array([-0.5, 2.0, -0.25])
    affine = AffineSubspaceProjector(np.array([[1.0, 0.0, 0.0]]), np.array([1.0]))
    candidate = DykstraProjector((affine, psd_cone_projector(2)))
    projected = candidate.project(point)

    s = cp.Variable(3)
    X = cp.bmat([[s[0], s[1] / root2], [s[1] / root2, s[2]]])
    problem = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(s - point)),
                         [s[0] == 1.0, X >> 0])
    problem.solve(solver=cp.CLARABEL, tol_gap_abs=1e-11,
                  tol_gap_rel=1e-11, tol_feas=1e-11)
    # Clarabel's semidefinite residual at this rank-one boundary is around
    # 6e-7 even with its tight tolerances; the Dykstra point itself settles to
    # 1e-12 in both the affine and PSD residuals.
    np.testing.assert_allclose(projected, s.value, atol=2e-6)


def test_dykstra_candidate_gradient_map_certificate_passes_and_nonoptimal_fails():
    candidate = joint_dykstra_projector(
        np.array([[-1.0, 0.0]]), np.array([0.5]),
        cones=(),
    )
    problem = OptimizationProblem(
        np.eye(2), np.array([-1.0, -0.2]),
        np.zeros((0, 2)), np.zeros(0), nonlinear_candidates=(candidate,),
    )
    solver = SNNSolver(problem, SolverConfig(
        k0=0.5, max_iterations=400,
        convergence=ConvergenceConfig(enable_early_stopping=False),
    ))
    optimum = candidate.project(np.array([1.0, 0.2]))
    cert_opt = solver._compute_kkt_certificate(optimum)
    assert cert_opt.passed
    # This point is feasible for the halfspace but is not optimal for the
    # objective, so the exact-set gradient map must retain a defect.
    cert_bad = solver._compute_kkt_certificate(np.array([0.5, 0.0]))
    assert not cert_bad.passed


def _ball_halfspace_reference(point, radius=1.0, upper=0.5):
    """Closed-form projection onto a ball intersected with x[0] <= upper."""
    point = np.asarray(point, dtype=float)
    norm = np.linalg.norm(point)
    ball_point = point if norm <= radius else radius * point / norm
    if ball_point[0] <= upper:
        return ball_point
    radial = point[1:]
    radial_norm = np.linalg.norm(radial)
    radial_radius = np.sqrt(radius * radius - upper * upper)
    if radial_norm <= radial_radius:
        radial_point = radial
    else:
        radial_point = radial_radius * radial / radial_norm
    return np.concatenate(([upper], radial_point))


def _lens_reference(point, center_x, radius=0.7):
    """Projection onto two balls whose centers lie on the first axis."""
    point = np.asarray(point, dtype=float)
    center = np.array([center_x, 0.0, 0.0])
    norm = np.linalg.norm(point)
    first = point if norm <= 1.0 else point / norm
    if np.linalg.norm(first - center) <= radius:
        return first
    shifted = point - center
    shifted_norm = np.linalg.norm(shifted)
    second = point if shifted_norm <= radius else center + radius * shifted / shifted_norm
    if np.linalg.norm(second) <= 1.0:
        return second
    # Both balls are active.  Their intersection is a circle in the plane
    # x[0] = (d^2 + 1 - r^2)/(2d); match the input's transverse direction.
    x0 = (center_x * center_x + 1.0 - radius * radius) / (2.0 * center_x)
    transverse_radius = np.sqrt(1.0 - x0 * x0)
    transverse = point[1:]
    transverse_norm = np.linalg.norm(transverse)
    if transverse_norm == 0.0:
        transverse_point = np.array([transverse_radius, 0.0])
    else:
        transverse_point = transverse_radius * transverse / transverse_norm
    return np.concatenate(([x0], transverse_point))


def test_dykstra_far_halfspace_regression_never_accepts_wrong_corner():
    point = np.array([2.7e8, 7.8e5, 4.0])
    candidate = joint_dykstra_projector(
        np.array([[1.0, 0.0, 0.0]]), np.array([-0.5]),
        members=(ball_projector([0, 1, 2], 1.0),),
    )
    projected = np.asarray(candidate.project(point))
    reference = _ball_halfspace_reference(point)
    diagnostics = candidate.last_diagnostics
    assert diagnostics["cap_hit"] or np.linalg.norm(projected - reference) <= 1e-9
    if diagnostics["converged"]:
        assert not diagnostics["cap_hit"]


def test_dykstra_scoped_far_halfspace_regression_never_accepts_wrong_corner():
    point = np.array([2.7e8, 7.8e5, 4.0, 1e12])
    candidate = DykstraProjector((
        halfspace_projector([1.0], -0.5, coordinates=[0]),
        ball_projector([0, 1, 2], 1.0),
    ), coordinates=[0, 1, 2])
    projected = np.asarray(candidate.project(point))
    reference = _ball_halfspace_reference(point[:3])
    diagnostics = candidate.last_diagnostics
    assert diagnostics["cap_hit"] or np.linalg.norm(projected[:3] - reference) <= 1e-9
    np.testing.assert_array_equal(projected[3:], point[3:])


def test_dykstra_scoped_moderate_input_ignores_spectator_scale():
    expected = np.array([0.5, np.sqrt(0.75)])
    active_results = []
    statuses = []
    for spectator in (0.0, 1e6, 1e12):
        candidate = DykstraProjector((
            halfspace_projector([1.0, 0.0], -0.5, coordinates=[0, 1]),
            ball_projector([0, 1], 1.0),
        ), coordinates=[0, 1], tolerance=1e-12, max_iterations=100)
        projected = np.asarray(candidate.project(
            np.array([0.7, 1.0, spectator])))
        active_results.append(projected[:2])
        statuses.append(candidate.last_diagnostics["converged"])
        np.testing.assert_equal(projected[2], spectator)
    np.testing.assert_allclose(active_results, [expected] * 3, atol=1e-12)
    assert statuses == [True, True, True]


def test_dykstra_far_halfspace_sweep_is_exact_or_honest_cap():
    direction = np.array([1.0, 7.8e5 / 2.7e8, 0.0])
    values = [1e-3, 1.0, 1e3, 1e6, 1e9]
    for scale in values:
        point = np.array([scale, direction[1] * scale, 4.0])
        candidate = joint_dykstra_projector(
            np.array([[1.0, 0.0, 0.0]]), np.array([-0.5]),
            members=(ball_projector([0, 1, 2], 1.0),),
            max_iterations=400,
        )
        reference = _ball_halfspace_reference(point)
        projected = np.asarray(candidate.project(point))
        diagnostics = candidate.last_diagnostics
        if diagnostics["converged"]:
            np.testing.assert_allclose(projected, reference, atol=1e-9)
        else:
            assert diagnostics["cap_hit"]


@pytest.mark.parametrize("center_x", [1.60, 1.69])
def test_dykstra_far_lens_sweeps_are_exact_or_honest_caps(center_x):
    values = [1e-3, 1.0, 1e3, 1e6, 1e9]
    for scale in values:
        # A transverse far direction activates both lens boundaries and
        # exercises the correction-settling floor.
        point = np.array([0.0, scale, 4.0])
        candidate = DykstraProjector((
            ball_projector([0, 1, 2], 1.0),
            ball_projector([0, 1, 2], 0.7, center=[center_x, 0.0, 0.0]),
        ), max_iterations=400)
        projected = np.asarray(candidate.project(point))
        reference = _lens_reference(point, center_x)
        diagnostics = candidate.last_diagnostics
        if diagnostics["converged"]:
            np.testing.assert_allclose(projected, reference, atol=1e-9)
        else:
            assert diagnostics["cap_hit"]


def test_dykstra_box_rows_projection_matches_clarabel():
    cp = pytest.importorskip("cvxpy")
    rng = np.random.default_rng(11)
    n = 6
    C = rng.standard_normal((3, n))
    d = -np.abs(rng.standard_normal(3))
    lower = np.array([-0.5, -1.0, 0.0, -np.inf, -0.2, -0.7])
    upper = np.array([0.6, 0.3, np.inf, 0.4, 0.2, 0.7])
    x0 = 3.0 * rng.standard_normal(n)
    candidate = joint_dykstra_projector(C, d, members=(box_projector(lower, upper),))
    projected = candidate.project(x0)
    x = cp.Variable(n)
    finite_lo = np.isfinite(lower)
    finite_hi = np.isfinite(upper)
    problem = cp.Problem(
        cp.Minimize(0.5 * cp.sum_squares(x - x0)),
        [C @ x + d <= 0, x[finite_lo] >= lower[finite_lo], x[finite_hi] <= upper[finite_hi]],
    )
    problem.solve(solver=cp.CLARABEL, tol_gap_abs=1e-11,
                  tol_gap_rel=1e-11, tol_feas=1e-11)
    np.testing.assert_allclose(projected, x.value, atol=2e-8)
    assert candidate.last_diagnostics["converged"]
    assert candidate.last_diagnostics["member_names"][-1] == "box"


def test_joint_dykstra_sparse_c_names_the_unsupported_setting():
    import scipy.sparse as sp

    from snn_opt import joint_dykstra_projector
    with pytest.raises(ValueError, match="scipy sparse C is not supported"):
        joint_dykstra_projector(sp.csr_matrix(np.ones((1, 3))), np.array([-1.0]))


def test_dykstra_cutter_member_keeps_precise_message():
    from snn_opt import DykstraProjector, spectral_ball_cutter
    with pytest.raises(TypeError, match="must be a ProjectorCandidate or callable"):
        DykstraProjector([spectral_ball_cutter(2)])
    with pytest.raises(TypeError, match="must be an iterable"):
        DykstraProjector(5)

"""Exact extended-certificate checks on smooth and nonsmooth faces."""

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
    joint_dykstra_projector,
    scaled_soc_projector,
    soc_projector,
    spectral_ball_projector,
    spectral_norm_cutter,
)


def _cert(A, b, candidate, x, **kwargs):
    problem = OptimizationProblem(
        np.asarray(A, dtype=float), np.asarray(b, dtype=float),
        np.zeros((0, len(x))), np.zeros(0), nonlinear_candidates=(candidate,))
    solver = SNNSolver(problem, SolverConfig(max_iterations=1, **kwargs))
    return solver._compute_kkt_certificate(np.asarray(x, dtype=float))


def test_ball_ray_is_kept_just_inside_and_tangent_fails():
    x = np.array([1.0 - 3e-9, 0.0, 0.0])
    ball = ball_projector([0, 1, 2], 1.0)
    assert ball.normal(x) is not None
    optimum = _cert(np.eye(3), -np.array([2.0, 0.0, 0.0]), ball, x)
    assert optimum.passed
    tangent = _cert(np.eye(3), np.array([0.0, 1.0, 0.0]) - x, ball,
                    x)
    assert not tangent.passed


def test_soc_and_scaled_soc_apex_use_the_full_polar():
    z = np.array([-1.0, 0.3, -0.2])
    soc = soc_projector(0, [1, 2])
    apex = _cert(np.eye(3), -z, soc, np.zeros(3))
    assert apex.passed
    nonoptimal = _cert(np.eye(3), np.array([-0.4, 0.1, 0.0]), soc,
                       np.zeros(3))
    assert not nonoptimal.passed

    mu = 0.4
    polar = np.array([-1.0, 0.1, 0.05])
    scaled = scaled_soc_projector(0, [1, 2], mu)
    assert np.allclose(scaled.project(polar), 0.0)
    scaled_cert = _cert(np.eye(3), -polar, scaled, np.zeros(3))
    assert scaled_cert.passed

    scaled_nonoptimal = _cert(
        np.eye(3), np.array([-0.4, 0.1, 0.0]), scaled, np.zeros(3))
    assert not scaled_nonoptimal.passed


def test_near_apex_polar_model_pays_local_complementarity():
    soc = soc_projector(0, [1, 2])
    x = np.array([1e-3, 0.0, 0.0])
    # The point is within the apex activity band, but it is an interior point
    # of the SOC and the nonzero gradient is therefore non-optimal.  A free
    # polar model would cancel it exactly; local complementarity must reject it.
    gradient = np.array([1.0, 0.0, 0.0])
    cert = _cert(np.eye(3), gradient - x, soc, x)
    assert not cert.passed
    assert cert.complementarity >= 9e-4


def test_smooth_soc_boundary_with_tangent_gradient_stays_rejected():
    soc = soc_projector(0, [1, 2, 3])
    x = np.array([1.0, 1.0, 0.0, 0.0])
    gradient = np.array([0.0, 0.0, 1e-3, 0.0])
    cert = _cert(np.eye(4), gradient - x, soc, x)
    assert not cert.passed


def test_cutter_normal_stack_certifies_a_corner():
    def value(x):
        return float(max(x[0] ** 2, x[1] ** 2) - 1.0)

    def jacobian(x):
        return np.array([2.0 * x[0], 0.0, 0.0]) if x[0] ** 2 >= x[1] ** 2 \
            else np.array([0.0, 2.0 * x[1], 0.0])

    def normal(x):
        return np.array([[2.0 * x[0], 0.0, 0.0],
                         [0.0, 2.0 * x[1], 0.0]])

    def piece_slacks(x):
        return np.array([
            (1.0 - x[0] ** 2) / max(2.0 * abs(x[0]), 1e-12),
            (1.0 - x[1] ** 2) / max(2.0 * abs(x[1]), 1e-12),
        ])

    cutter = CutterCandidate(value, jacobian, normal=normal,
                             kkt_data={"slack": piece_slacks})
    cert = _cert(np.eye(3), -np.array([3.0, 3.0, 0.0]), cutter,
                 np.array([1.0, 1.0, 0.0]))
    assert cert.passed


def test_cutter_normal_stack_requires_piece_slacks_off_kink():
    def value(x):
        return float(max(x[0] ** 2, x[1] ** 2) - 1.0)

    def jacobian(x):
        return np.array([2.0 * x[0], 0.0, 0.0]) if x[0] ** 2 >= x[1] ** 2 \
            else np.array([0.0, 2.0 * x[1], 0.0])

    def normal(x):
        return np.array([[2.0 * x[0], 0.0, 0.0],
                         [0.0, 2.0 * x[1], 0.0]])

    x = np.array([1.0, 0.2, 0.0])
    target = np.array([2.0, 1.2, 0.0])
    missing = CutterCandidate(value, jacobian, normal=normal)
    unknown = _cert(np.eye(3), -target, missing, x)
    assert unknown.fit_status == "not_available"
    assert not unknown.passed

    def piece_slacks(y):
        return np.array([
            (1.0 - y[0] ** 2) / max(2.0 * abs(y[0]), 1e-12),
            (1.0 - y[1] ** 2) / max(2.0 * abs(y[1]), 1e-12),
        ])
    priced = CutterCandidate(value, jacobian, normal=normal,
                             kkt_data={"slack": piece_slacks})
    cert = _cert(np.eye(3), -target, priced, x)
    assert cert.fit_status == "ok"
    assert not cert.passed
    assert cert.residual > cert.tolerance

    scalar = CutterCandidate(value, jacobian, normal=normal,
                             kkt_data={"slack": lambda y: 0.0})
    scalar_cert = _cert(np.eye(3), -target, scalar, x)
    assert scalar_cert.fit_status == "not_available"


def test_exact_projector_certificate_is_independent_of_solver_step_scale():
    def project(x):
        y = np.asarray(x, dtype=float).copy()
        norm = np.linalg.norm(y)
        if norm > 1.0:
            y *= 1.0 / norm
        return y

    candidate = ProjectorCandidate(
        project=project, name="unit_ball_hook",
        kkt_data={"euclidean_project": project},
    )
    x = np.zeros(3)
    optimum = np.array([1.0, 0.0, 0.0])
    for k0 in (1e-3, 0.5, 1e4):
        cert = _cert(np.eye(3), -np.array([3e4, 0.0, 0.0]), candidate,
                     x, k0=k0)
        assert not cert.passed
        assert cert.fit_status == "ok"
        assert cert.residual > 1e3 * cert.tolerance
        optimum_cert = _cert(np.eye(3), -np.array([3e4, 0.0, 0.0]),
                             candidate, optimum, k0=k0)
        assert optimum_cert.passed


def test_seed3_rows_soc_clarabel_optima_all_certify():
    cp = pytest.importorskip("cvxpy")
    rng = np.random.default_rng(3)
    for _ in range(30):
        matrix = rng.standard_normal((4, 4))
        A = matrix.T @ matrix + 0.5 * np.eye(4)
        b = rng.standard_normal(4) * 3.0
        C = rng.standard_normal((2, 4))
        x_var = cp.Variable(4)
        reference = cp.Problem(
            cp.Minimize(0.5 * cp.quad_form(x_var, A) + b @ x_var),
            [C @ x_var <= 0, cp.norm(x_var[1:]) <= x_var[0]],
        )
        reference.solve(solver=cp.CLARABEL, tol_gap_abs=1e-10,
                        tol_gap_rel=1e-10, tol_feas=1e-10)
        assert x_var.value is not None
        candidate = soc_projector(0, [1, 2, 3])
        problem = OptimizationProblem(
            A, b, C, np.zeros(2), nonlinear_candidates=(candidate,))
        solver = SNNSolver(problem, SolverConfig(max_iterations=1))
        certificate = solver._compute_kkt_certificate(x_var.value)
        assert certificate.passed, (
            f"Clarabel optimum did not certify: status={certificate.fit_status}, "
            f"residual={certificate.residual}, tolerance={certificate.tolerance}")


def test_spectral_projector_map_handles_ties_and_rejects_nonoptimal_points():
    cutter = spectral_norm_cutter((3, 3), 1.0)
    optimum = np.eye(3)
    cert = _cert(0.2 * np.eye(9), -optimum.ravel(), cutter, optimum.ravel())
    assert cert.passed

    nonoptimal = _cert(np.eye(9), np.zeros(9), cutter, optimum.ravel())
    assert not nonoptimal.passed

    def value(x):
        return float(np.linalg.svd(np.asarray(x).reshape(3, 3),
                                   compute_uv=False)[0] - 1.0)

    def jacobian(x):
        u, _, vt = np.linalg.svd(np.asarray(x).reshape(3, 3))
        return np.outer(u[:, 0], vt[0]).ravel()

    one_row = CutterCandidate(value, jacobian)
    one_row_cert = _cert(0.2 * np.eye(9), -optimum.ravel(), one_row,
                         optimum.ravel())
    assert not one_row_cert.passed


def test_unknown_projector_stays_not_available():
    unknown = ProjectorCandidate(project=lambda x: np.asarray(x), name="unknown")
    cert = _cert(np.eye(2), np.zeros(2), unknown, np.zeros(2))
    assert cert.fit_status == "not_available"
    assert not cert.passed


def test_spectral_tie_with_active_upper_bound_is_jointly_fit():
    x = np.eye(2).ravel()
    cutter = spectral_norm_cutter((2, 2), 1.0)
    cert = _cert(0.1 * np.eye(4), -np.eye(2).ravel(), cutter, x,
                 upper_bound=1.0)
    assert cert.passed


def _ill_conditioned_spectral(scale=1.0):
    """min 1/2||X||_F^2 - <G, X> on ||X||_2 <= 1, G = I + diag(2.7e8, 7.8e5, 4).

    X* = I. A feasible diagonal point diag(1-d, 1, 1) leaves the stiff mode
    slack by d while the other two singular values stay saturated.
    """
    stiff = np.array([2.7e8, 7.8e5, 4.0])
    goal = np.diag(1.0 + stiff)
    return (scale * np.eye(9), -scale * goal.ravel(),
            spectral_norm_cutter(3), stiff)


def test_ill_conditioned_spectral_near_optimum_certifies():
    """A face gap of a few 1e-7 must certify. The old fixed 1e-5 state step
    turned that gap into residual/tol ~ 1e5 * d and rejected d = 3e-9.
    """
    A, b, cutter, _stiff = _ill_conditioned_spectral()
    for d in (0.0, 1e-9, 3e-9, 3e-8, 3e-7):
        X = np.diag([1.0 - d, 1.0, 1.0])
        cert = _cert(A, b, cutter, X.ravel())
        assert cert.fit_status == "ok"
        assert cert.passed, (
            f"d={d}: residual/tol={cert.residual / cert.tolerance}")
    d = 3e-7
    X = np.diag([1.0 - d, 1.0, 1.0])
    cert = _cert(A, b, cutter, X.ravel())
    # With A = I, the full curvature step projects directly to X*. The
    # certificate is the state gap d, independent of the normal magnitude.
    assert cert.complementarity == 0.0
    assert cert.stationarity == pytest.approx(d, rel=1e-8)
    assert cert.scale == pytest.approx(np.linalg.norm(X))
    reference = cert.residual
    for k0 in (1e-3, 1e4):
        scaled = _cert(A, b, cutter, X.ravel(), k0=k0)
        assert scaled.passed
        assert scaled.residual == pytest.approx(reference, rel=1e-8)


def test_ill_conditioned_spectral_nonstationary_points_fail():
    """Order-one state gaps must fail regardless of the normal magnitude."""
    A, b, cutter, _stiff = _ill_conditioned_spectral()
    interior = _cert(A, b, cutter, (0.5 * np.eye(3)).ravel())
    assert interior.fit_status == "ok"
    assert not interior.passed
    assert interior.stationarity > interior.tolerance

    quarter = np.array([
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    rotated = _cert(A, b, cutter, quarter.ravel())
    assert rotated.fit_status == "ok"
    assert not rotated.passed
    assert rotated.stationarity > rotated.tolerance


def test_ill_conditioned_spectral_decision_is_invariant_under_objective_scaling():
    near_x = np.diag([1.0 - 3e-7, 1.0, 1.0]).ravel()
    bad_x = (0.5 * np.eye(3)).ravel()
    near_ratios = []
    bad_ratios = []
    for scale in (1e-6, 1.0, 1e6):
        A, b, cutter, _stiff = _ill_conditioned_spectral(scale)
        near = _cert(A, b, cutter, near_x)
        bad = _cert(A, b, cutter, bad_x)
        assert near.fit_status == "ok" and near.passed, scale
        assert bad.fit_status == "ok" and not bad.passed, scale
        near_ratios.append(near.residual / near.tolerance)
        bad_ratios.append(bad.residual / bad.tolerance)
    assert near_ratios[0] == pytest.approx(near_ratios[1], rel=1e-6)
    assert near_ratios[1] == pytest.approx(near_ratios[2], rel=1e-6)
    assert bad_ratios[0] == pytest.approx(bad_ratios[1], rel=1e-6)
    assert bad_ratios[1] == pytest.approx(bad_ratios[2], rel=1e-6)


def test_positive_objective_scaling_preserves_apex_and_spectral_certificates():
    soc = soc_projector(0, [1, 2])
    apex_gradient = np.array([1.0, -0.3, 0.2])
    spectral = spectral_norm_cutter((2, 2), 1.0)
    spectral_x = np.eye(2).ravel()
    for scale in (1e-6, 1e6):
        apex = _cert(scale * np.eye(3), scale * apex_gradient,
                     soc, np.zeros(3))
        assert apex.passed
        spectral_cert = _cert(
            scale * 0.2 * np.eye(4), -scale * spectral_x,
            spectral, spectral_x)
        assert spectral_cert.passed


def test_rows_plus_soc_apex_cold_solve_matches_clarabel_point():
    # Same seed and instance as verify_joint.py, with the Clarabel optimum
    # recorded inline so this regression does not import a second solver.
    rng = np.random.default_rng(3)
    for _ in range(3):
        n = 4
        matrix = rng.standard_normal((n, n))
        A = matrix.T @ matrix + 0.5 * np.eye(n)
        b = rng.standard_normal(n) * 3.0
        C = rng.standard_normal((2, n))
    clarabel_x = np.array([0.0, 0.0, 0.0, 0.0])
    problem = OptimizationProblem(
        A, b, C, np.zeros(2),
        nonlinear_candidates=(soc_projector(0, [1, 2, 3]),))
    result = SNNSolver(problem, SolverConfig(max_iterations=5000)).solve(
        np.zeros(n))
    assert result.converged
    assert np.linalg.norm(result.X[-1] - clarabel_x) <= 1e-5


def test_stiff_dykstra_corner_certifies_true_optimum_not_long_step_artifact():
    # ball ∩ {x1 <= 0.5} under a stiff gradient. Projecting a point ||g||/L
    # away with Dykstra "converges" (input-scaled tolerance) to a wrong
    # corner; the capped trial step keeps the certificate on the true one.
    from snn_opt import ball_projector, joint_dykstra_projector
    g = np.array([2.7e8, 7.8e5, 4.0])
    tail = g[1:] / np.linalg.norm(g[1:])
    x_true = np.array([0.5, *(np.sqrt(0.75) * tail)])

    def certificate(x):
        dyk = joint_dykstra_projector(np.array([[1.0, 0.0, 0.0]]),
                                      np.array([-0.5]),
                                      members=[ball_projector([0, 1, 2], 1.0)])
        prob = OptimizationProblem(np.eye(3), -g, np.zeros((0, 3)), np.zeros(0),
                                   nonlinear_candidates=(dyk,))
        return SNNSolver(prob, SolverConfig())._compute_kkt_certificate(x)

    for delta in (0.0, 3e-9, 3e-7):
        cert = certificate(x_true - delta * np.array([1.0, 0.0, 0.0]))
        assert cert.fit_status == "ok" and cert.passed, (delta, cert)
    wrong_corner = np.array([1.28205046e-06, 1.0, 5.12820513e-06])
    assert not certificate(wrong_corner).passed


def _idx6_solver(A=None, upper_bound=None):
    """The stored 06a index-6 instance with its ten-step cold oracle."""
    Hn = np.array([
        [0.30837371653256174, -0.059593763558365306, 0.240964926711467],
        [-0.6997078519533171, 0.1352110157543459, -0.5467072402631933],
        [0.14888658938478824, -0.0287847707756936, 0.11632908252837862],
    ])
    exact_optimum = np.array([
        [-0.26223457102305564, -0.1105449143426895, 0.9586515799148935],
        [-0.957757836170077, -0.09169897237004083, -0.2725641680799044],
        [0.11803794735239126, -0.9896318105130039, -0.08182861727774811],
    ]).ravel()
    eps = 5.140367878587841e-6

    def top(x):
        M = np.asarray(x).reshape(3, 3)
        v = np.ones(3) / np.sqrt(3.0)
        for _ in range(10):
            w = M.T @ (M @ v)
            norm = np.linalg.norm(w)
            if norm == 0.0:
                break
            v = w / norm
        Mv = M @ v
        sigma = np.linalg.norm(Mv)
        u = Mv / sigma if sigma > 0.0 else np.array([1.0, 0.0, 0.0])
        return sigma, u, v

    cutter = CutterCandidate(
        value=lambda x: float(top(x)[0] - 1.0),
        jacobian=lambda x: np.outer(top(x)[1], top(x)[2]).ravel(),
        kkt_data={"euclidean_project": spectral_ball_projector((3, 3), 1.0).project},
    )
    problem = OptimizationProblem(
        np.eye(9) if A is None else A, -Hn.ravel() / eps,
        np.zeros((0, 9)), np.zeros(0),
        nonlinear_candidates=(cutter,))
    solver = SNNSolver(problem, SolverConfig(
        k0=0.5, max_iterations=200, constraint_tol=1e-10,
        upper_bound=upper_bound,
        convergence=ConvergenceConfig(check_every=1, min_iterations=1, patience=1)))
    return solver, exact_optimum


def test_inexact_power_oracle_cannot_certify_a_wrong_spectral_rotation():
    # The cold ten-step oracle stalls ~0.026 away from the polar optimum.
    # A gradient-relative tolerance of ~19 used to certify at iteration 11.
    solver, exact_optimum = _idx6_solver()
    result = solver.solve(np.zeros(9))
    assert not result.converged
    assert result.kkt_residual > result.kkt_tolerance
    assert np.linalg.norm(result.final_x - exact_optimum) > 1e-3
    assert solver._compute_kkt_certificate(exact_optimum).passed


def test_ulp_asymmetry_keeps_the_idx6_state_certificate():
    A = np.eye(9)
    A[0, 1] += np.finfo(float).eps
    solver, exact_optimum = _idx6_solver(A=A)
    result = solver.solve(np.zeros(9))
    assert not result.converged
    assert np.linalg.norm(result.final_x - exact_optimum) > 1e-3
    assert solver._strong_convexity() > 0.0

    from scipy.sparse import eye, lil_matrix
    sparse_A = lil_matrix(eye(9))
    sparse_A[0, 1] += np.finfo(float).eps
    sparse_problem = solver.problem.__class__(
        sparse_A.tocsr(), solver.problem.b, solver.problem.C, solver.problem.d,
        nonlinear_candidates=solver.problem.nonlinear_candidates)
    sparse_solver = SNNSolver(sparse_problem, SolverConfig())
    assert sparse_solver._strong_convexity() > 0.0


def test_near_symmetry_deflation_keeps_the_idx6_state_certificate():
    # A perturbation just beyond the old hard symmetry window must still use
    # the symmetric-part state bound.  Deflating mu by the skew budget keeps
    # the contraction valid and rejects the cold-power stall.
    A = np.eye(9)
    A[0, 1] += 1.615e-14
    solver, exact_optimum = _idx6_solver(A=A)
    result = solver.solve(np.zeros(9))
    assert not result.converged
    assert np.linalg.norm(result.final_x - exact_optimum) > 1e-3
    assert solver._strong_convexity() > 0.0


def test_inactive_box_facets_use_the_state_certificate():
    solver, exact_optimum = _idx6_solver(upper_bound=1e6)
    result = solver.solve(np.zeros(9))
    assert not result.converged
    assert np.linalg.norm(result.final_x - exact_optimum) > 1e-3

    # An actually binding box remains on the ordinary joint row fit.
    x = np.eye(2).ravel()
    candidate = spectral_norm_cutter((2, 2), 1.0)
    bound_problem = OptimizationProblem(
        np.eye(4), -x, np.zeros((0, 4)), np.zeros(0),
        nonlinear_candidates=(candidate,))
    bound_solver = SNNSolver(bound_problem, SolverConfig(upper_bound=1.0))
    bound_cert = bound_solver._compute_kkt_certificate(x)
    assert bound_cert.fit_status == "ok" and bound_cert.passed


@pytest.mark.parametrize("scale", [1e-6, 1.0, 1e6])
@pytest.mark.parametrize("sparse", [False, True])
def test_exact_projector_state_bound_uses_smallest_curvature(scale, sparse):
    # The large gradient is normal to x[0] <= 1. The state error lies in
    # the weakly curved free coordinate, so L alone is not an error bound.
    A = scale * np.diag([1.0, 1e-4])
    if sparse:
        from scipy.sparse import csr_matrix
        A = csr_matrix(A)

    def project(x):
        out = np.asarray(x).copy()
        out[0] = min(1.0, out[0])
        return out

    candidate = ProjectorCandidate(
        project, coordinates=(0,), kkt_data={"euclidean_project": project})
    problem = OptimizationProblem(
        A, scale * np.array([-1e6, 0.0]), np.zeros((0, 2)), np.zeros(0),
        nonlinear_candidates=(candidate,))
    solver = SNNSolver(problem, SolverConfig())
    bad = solver._compute_kkt_certificate(np.array([1.0, 0.01]))
    assert bad.fit_status == "ok" and not bad.passed
    assert bad.residual == pytest.approx(0.01, rel=1e-8)
    assert solver._compute_kkt_certificate(np.array([1.0, 0.0])).passed


def test_rank_deficient_exact_projector_keeps_gradient_unit_certificate():
    """Merely convex objectives do not receive a spurious state bound."""
    candidate = spectral_norm_cutter((2, 2), 1.0)
    problem = OptimizationProblem(
        np.diag([1.0, 0.0, 0.0, 0.0]),
        np.array([-1e6, -2e5, 0.0, 0.0]),
        np.zeros((0, 4)), np.zeros(0), nonlinear_candidates=(candidate,))
    solver = SNNSolver(problem, SolverConfig())
    cert = solver._compute_kkt_certificate(np.eye(2).ravel())
    # The rank-deficient Hessian follows the ordinary gradient-unit path;
    # its scale is objective-sized rather than max(1, ||x||).
    assert solver._strong_convexity() == 0.0
    assert cert.scale > 1e5
    assert cert.tolerance > 1.0


def test_sparse_graded_minimum_eigenvalue_resolves_for_state_certificate():
    from scipy.sparse import diags
    n = 40
    A = diags(np.geomspace(1e-6, 1.0, n), format="csr")

    def project(x):
        arr = np.asarray(x, dtype=float)
        norm = np.linalg.norm(arr)
        return arr if norm <= 1e6 else arr * (1e6 / norm)

    candidate = ProjectorCandidate(
        project, coordinates=tuple(range(n)),
        kkt_data={"euclidean_project": project})
    problem = OptimizationProblem(
        A, np.zeros(n), np.zeros((0, n)), np.zeros(0),
        nonlinear_candidates=(candidate,))
    solver = SNNSolver(problem, SolverConfig())
    x = np.zeros(n)
    x[0] = 1e-3
    cert = solver._compute_kkt_certificate(x)
    assert solver._strong_convexity() > 0.0
    assert cert.fit_status == "ok" and not cert.passed
    assert cert.residual == pytest.approx(1e-3, rel=2e-5)


@pytest.mark.parametrize("diagonal", [
    np.array([-100.0, 1.0]),
    np.array([-10.0, 0.05, 1.0, 2.0]),
])
def test_sparse_indefinite_hessian_keeps_gradient_unit_certificate(diagonal):
    from scipy.sparse import diags
    candidate = ball_projector(tuple(range(diagonal.size)), 1.0)
    problem = OptimizationProblem(
        diags(diagonal, format="csr"), np.zeros(diagonal.size),
        np.zeros((0, diagonal.size)), np.zeros(0),
        nonlinear_candidates=(candidate,))
    solver = SNNSolver(problem, SolverConfig())
    x = np.zeros(diagonal.size)
    # Shift-invert previously assigned positive curvature and admitted this
    # nonstationary point near the saddle. The ordinary KKT test rejects it.
    x[0] = 2e-7
    cert = solver._compute_kkt_certificate(x)
    assert solver._strong_convexity() == 0.0
    assert cert.fit_status == "ok"
    assert cert.scale < 1e-3
    assert not cert.passed


@pytest.mark.parametrize("diagonal", [
    np.array([1.0, 0.0, 0.0, 0.0]),
    np.zeros(4),
])
def test_sparse_singular_hessian_keeps_gradient_unit_certificate(diagonal):
    from scipy.sparse import diags
    candidate = spectral_norm_cutter((2, 2), 1.0)
    problem = OptimizationProblem(
        diags(diagonal, format="csr"), np.array([-1e6, -2e5, 0.0, 0.0]),
        np.zeros((0, diagonal.size)), np.zeros(0),
        nonlinear_candidates=(candidate,))
    solver = SNNSolver(problem, SolverConfig())
    cert = solver._compute_kkt_certificate(np.eye(2).ravel())
    assert solver._strong_convexity() == 0.0
    assert cert.fit_status == "ok"
    assert cert.scale > 1e5
    assert cert.tolerance > 1.0


def test_sparse_curvature_solver_failure_fails_closed(monkeypatch):
    import scipy.sparse.linalg as sparse_linalg
    from scipy.sparse import eye
    A = eye(4, format="csr")
    def project(x):
        arr = np.asarray(x, dtype=float)
        norm = np.linalg.norm(arr)
        return arr if norm <= 10.0 else arr * (10.0 / norm)

    candidate = ProjectorCandidate(
        project, coordinates=tuple(range(4)),
        kkt_data={"euclidean_project": project})
    problem = OptimizationProblem(
        A, np.zeros(4), np.zeros((0, 4)), np.zeros(0),
        nonlinear_candidates=(candidate,))
    solver = SNNSolver(problem, SolverConfig())

    calls = []
    def fail(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("synthetic eigensolver failure")

    monkeypatch.setattr(sparse_linalg, "eigsh", fail)
    cert = solver._compute_kkt_certificate(np.array([1e-2, 0, 0, 0]))
    assert cert.fit_status == "fit_failed" and not cert.passed
    assert solver._compute_kkt_certificate(np.array([1e-2, 0, 0, 0])).fit_status == "fit_failed"
    assert calls == [1]
    assert solver._strong_convexity_cache == 0.0


def test_sparse_graded_arpack_fallback_is_conservative_and_cached(monkeypatch):
    import scipy.sparse.linalg as sparse_linalg
    from scipy.sparse import diags
    n = 200
    A = diags(np.geomspace(1e-6, 1.0, n), format="csr")

    def project(x):
        return np.asarray(x, dtype=float)

    candidate = ProjectorCandidate(
        project, coordinates=tuple(range(n)),
        kkt_data={"euclidean_project": project})
    solver = SNNSolver(OptimizationProblem(
        A, np.zeros(n), np.zeros((0, n)), np.zeros(0),
        nonlinear_candidates=(candidate,)), SolverConfig())
    original = sparse_linalg.eigsh
    calls = []

    def counted(*args, **kwargs):
        calls.append(kwargs.get("which"))
        return original(*args, **kwargs)

    monkeypatch.setattr(sparse_linalg, "eigsh", counted)
    zero = np.zeros(n)
    at_zero = solver._compute_kkt_certificate(zero)
    away = zero.copy()
    away[0] = 1e-3
    away_cert = solver._compute_kkt_certificate(away)
    assert solver._strong_convexity() == pytest.approx(1e-6, rel=1e-6)
    assert at_zero.fit_status == "ok" and at_zero.passed
    assert away_cert.fit_status == "ok" and not away_cert.passed
    assert calls.count("SA") == 1
    assert calls.count("LM") == 1


def test_dykstra_state_residual_uses_the_triangle_bound():
    dyk = joint_dykstra_projector(
        np.array([[1.0, 0.0]]), np.zeros(1), tolerance=1e-12)
    problem = OptimizationProblem(
        np.diag([1.0, 0.5]), np.zeros(2), np.zeros((0, 2)), np.zeros(0),
        nonlinear_candidates=(dyk,))
    solver = SNNSolver(problem, SolverConfig())
    cert = solver._compute_kkt_certificate(np.array([0.0, 1.2e-4]))
    assert cert.fit_status == "ok" and not cert.passed
    assert cert.residual == pytest.approx(
        cert.stationarity + cert.complementarity, rel=1e-12)


def test_loose_dykstra_tolerance_fails_closed():
    C = np.array([[1.0, 0.0], [np.cos(0.7), np.sin(0.7)]])
    dyk = joint_dykstra_projector(C, np.zeros(2), tolerance=1e-3)
    target = np.array([1.0, 0.5])
    problem = OptimizationProblem(
        np.eye(2), -target, np.zeros((0, 2)), np.zeros(0),
        nonlinear_candidates=(dyk,))
    solver = SNNSolver(problem, SolverConfig())
    x = dyk.project(target)
    cert = solver._compute_kkt_certificate(x)
    assert cert.fit_status == "fit_failed" and not cert.passed


@pytest.mark.parametrize(("dyk_tolerance", "fit_status", "passed"), [
    (1e-12, "ok", True),
    (1e-8, "ok", True),
    (1e-6, "fit_failed", False),
])
def test_stiff_dykstra_tolerance_margin_is_step_aware(
        dyk_tolerance, fit_status, passed):
    g = np.array([2.7e8, 7.8e5, 4.0])
    tail = g[1:] / np.linalg.norm(g[1:])
    x_true = np.array([0.5, *(np.sqrt(0.75) * tail)])
    dyk = joint_dykstra_projector(
        np.array([[1.0, 0.0, 0.0]]), np.array([-0.5]),
        members=[ball_projector([0, 1, 2], 1.0)], tolerance=dyk_tolerance)
    problem = OptimizationProblem(
        np.eye(3), -g, np.zeros((0, 3)), np.zeros(0),
        nonlinear_candidates=(dyk,))
    solver = SNNSolver(problem, SolverConfig())
    cert = solver._compute_kkt_certificate(x_true)
    assert cert.fit_status == fit_status
    assert cert.passed is passed


def test_dykstra_positional_gate_uses_full_state_tolerance():
    dyk = joint_dykstra_projector(
        np.eye(2), -0.5 * np.ones(2), tolerance=5e-5)
    problem = OptimizationProblem(
        np.eye(2), -np.ones(2), np.zeros((0, 2)), np.zeros(0),
        nonlinear_candidates=(dyk,))
    convergence = ConvergenceConfig(kkt_abs_tol=1e-2, kkt_rel_tol=1e-4)
    solver = SNNSolver(problem, SolverConfig(convergence=convergence))
    cert = solver._compute_kkt_certificate(np.array([0.5, 0.5]))
    assert cert.fit_status == "ok" and cert.passed


def test_tiny_curvature_roundoff_keeps_the_old_certificate_contract():
    mu = 1e-12
    A = np.diag([1.0, mu, mu, mu])
    Q = np.array([[0.8727445076457513, -0.4881772468829075],
                  [0.4881772468829075, 0.8727445076457513]])
    x = Q.ravel()
    b = (-Q).ravel() - A @ x
    candidate = spectral_ball_projector((2, 2), 1.0)
    problem = OptimizationProblem(
        A, b, np.zeros((0, 4)), np.zeros(0),
        nonlinear_candidates=(candidate,))
    cert = SNNSolver(problem, SolverConfig())._compute_kkt_certificate(x)
    assert cert.fit_status == "ok" and cert.passed

"""Parity coverage for the native backend's built-in conic candidates.

The released polyhedral parity battery remains in ``test_c_backend_parity``.
This module covers the descriptor-driven entry point added for Round B.
"""

from __future__ import annotations

import numpy as np
import pytest

from snn_opt import (
    AffineSubspaceProjector,
    ConvergenceConfig,
    CutterCandidate,
    DykstraProjector,
    OptimizationProblem,
    ProjectorCandidate,
    SNNSolver,
    SolverConfig,
    ball_projector,
    halfspace_projector,
    joint_dykstra_projector,
    lift_soc_l1,
    lift_soc_l2,
    psd_cone_projector,
    scaled_soc_projector,
    soc_projector,
    spectral_ball_projector,
)

_kernel = pytest.importorskip(
    "snn_opt._kernel",
    reason="compiled C++ kernel not built (python setup.py build_ext --inplace)",
)


def _pair(problem, x0, *, max_iterations=8, convergence=None, max_projection_iters=None):
    """Solve one problem through Python and C and assert the parity contract."""
    conv_kwargs = dict(convergence or {})
    results = {}
    for backend in ("python", "c"):
        cfg_kwargs = {
            "backend": backend,
            "max_iterations": max_iterations,
            "record_trajectory": backend == "python",
            "convergence": ConvergenceConfig(**conv_kwargs),
        }
        if max_projection_iters is not None:
            cfg_kwargs["max_projection_iters"] = max_projection_iters
        results[backend] = SNNSolver(problem, SolverConfig(**cfg_kwargs)).solve(x0)

    py = results["python"]
    native = results["c"]
    dx = float(np.max(np.abs(py.final_x - native.final_x)))
    assert dx < 1e-7, f"final_x diverged by {dx:.2e}"
    assert native.converged == py.converged
    assert native.iterations_used == py.iterations_used
    assert native.spike_event_kinds == py.spike_event_kinds
    np.testing.assert_array_equal(native.spike_event_indices, py.spike_event_indices)
    assert native.nonlinear_event_counts == py.nonlinear_event_counts
    return py, native


def _fixed_fixtures():
    return [
        pytest.param(
            "ball-n2",
            OptimizationProblem(
                np.eye(2),
                np.zeros(2),
                np.zeros((0, 2)),
                np.zeros(0),
                nonlinear_candidates=(ball_projector([0, 1], 1.0),),
            ),
            np.array([2.0, -0.5]),
            id="ball-n2",
        ),
        pytest.param(
            "ball-n5",
            OptimizationProblem(
                np.eye(5),
                np.zeros(5),
                np.zeros((0, 5)),
                np.zeros(0),
                nonlinear_candidates=(ball_projector([0, 1, 2, 3, 4], 0.8),),
            ),
            np.array([1.1, -0.4, 0.3, 0.7, -1.5]),
            id="ball-n5",
        ),
        pytest.param(
            "soc-boundary",
            OptimizationProblem(
                np.eye(3),
                np.zeros(3),
                np.zeros((0, 3)),
                np.zeros(0),
                nonlinear_candidates=(soc_projector(0, [1, 2]),),
            ),
            np.array([-0.2, 1.4, -0.2]),
            id="soc-boundary",
        ),
        pytest.param(
            "soc-apex",
            OptimizationProblem(
                np.eye(3),
                np.zeros(3),
                np.zeros((0, 3)),
                np.zeros(0),
                nonlinear_candidates=(soc_projector(0, [1, 2]),),
            ),
            np.array([-1.0, 0.3, -0.2]),
            id="soc-apex",
        ),
        pytest.param(
            "scaled-soc-apex",
            OptimizationProblem(
                np.eye(3),
                np.zeros(3),
                np.zeros((0, 3)),
                np.zeros(0),
                nonlinear_candidates=(scaled_soc_projector(0, [1, 2], 0.4),),
            ),
            np.array([-1.0, 0.3, -0.2]),
            id="scaled-soc-apex",
        ),
        pytest.param(
            "affine",
            OptimizationProblem(
                np.eye(2),
                np.zeros(2),
                np.zeros((0, 2)),
                np.zeros(0),
                nonlinear_candidates=(
                    AffineSubspaceProjector(
                        np.array([[1.0, 2.0], [3.0, -1.0]]), np.array([0.2, -0.3])
                    ),
                ),
            ),
            np.array([1.0, 2.0]),
            id="affine",
        ),
        pytest.param(
            "halfspace",
            OptimizationProblem(
                np.eye(2),
                np.zeros(2),
                np.zeros((0, 2)),
                np.zeros(0),
                nonlinear_candidates=(halfspace_projector(np.array([1.0, -2.0]), 0.3),),
            ),
            np.array([1.0, 0.0]),
            id="halfspace",
        ),
        pytest.param(
            "rows-soc",
            OptimizationProblem(
                np.eye(3),
                np.zeros(3),
                np.array([[1.0, 0.0, 0.0]]),
                np.array([-0.2]),
                nonlinear_candidates=(soc_projector(0, [1, 2]),),
            ),
            np.array([-0.3, 2.0, -1.0]),
            id="rows-soc",
        ),
        pytest.param(
            "rows-soc-dykstra",
            OptimizationProblem(
                np.eye(3),
                np.zeros(3),
                np.zeros((0, 3)),
                np.zeros(0),
                nonlinear_candidates=(
                    joint_dykstra_projector(
                        np.array([[-1.0, 0.0, 0.0]]),
                        np.array([0.5]),
                        cones=(soc_projector(0, [1, 2]),),
                    ),
                ),
            ),
            np.array([-0.3, 2.0, -1.0]),
            id="rows-soc-dykstra",
        ),
    ]


@pytest.mark.parametrize("name,problem,x0", _fixed_fixtures())
def test_builtin_conic_fixture_parity(name, problem, x0):
    del name
    _pair(problem, x0, convergence={"enable_early_stopping": False})


def _friction_problem(seed=1000, q=4, mu=0.2):
    rng = np.random.default_rng(seed)
    A = np.zeros((3 * q, 3 * q))
    desired = []
    for i in range(q):
        R = rng.standard_normal((3, 3))
        A[3 * i : 3 * i + 3, 3 * i : 3 * i + 3] = R.T @ R + 0.5 * np.eye(3)
        d = rng.standard_normal(3)
        d[0] = abs(d[0]) + 1.0
        desired.append(d)
    desired = np.concatenate(desired)
    b = -A @ desired
    candidates = tuple(
        scaled_soc_projector(3 * i, [3 * i + 1, 3 * i + 2], mu, name=f"friction[{i}]")
        for i in range(q)
    )
    return OptimizationProblem(
        A, b, np.zeros((0, 3 * q)), np.zeros(0), nonlinear_candidates=candidates
    )


def test_scaled_soc_friction_blocks_q4_parity():
    _pair(
        _friction_problem(),
        np.zeros(12),
        max_iterations=60,
        convergence={"enable_early_stopping": False},
    )


@pytest.mark.parametrize("builder", [lift_soc_l1, lift_soc_l2], ids=["lift-soc-l1", "lift-soc-l2"])
def test_soc_lift_parity(builder):
    A = np.array([[3.0, 0.2], [0.2, 2.0]])
    b = np.array([-0.7, 0.4])
    K = np.array([[1.0, -0.5], [0.2, 0.8]])
    c = np.array([0.1, -0.2])
    lifted = builder(A, b, K, c, np.array([0.5, -0.3]), 0.9)
    _pair(
        lifted.problem,
        np.zeros(lifted.problem.n_vars),
        max_iterations=50,
        convergence={"enable_early_stopping": False},
    )


def test_affine_plus_rows_parity():
    problem = OptimizationProblem(
        np.eye(2),
        np.array([-1.0, 0.5]),
        np.array([[1.0, -2.0]]),
        np.array([-0.1]),
        nonlinear_candidates=(
            AffineSubspaceProjector(np.array([[1.0, 2.0], [3.0, -1.0]]), np.array([0.2, -0.3])),
        ),
    )
    _pair(problem, np.array([1.0, 2.0]), convergence={"enable_early_stopping": False})


def test_rows_soc_dykstra_apex_chunked_stop_parity():
    rng = np.random.default_rng(3)
    for _ in range(3):
        n = 4
        M = rng.standard_normal((n, n))
        A = M.T @ M + 0.5 * np.eye(n)
        b = rng.standard_normal(n) * 3
        C = rng.standard_normal((2, n))
    problem = OptimizationProblem(
        A,
        b,
        np.zeros((0, n)),
        np.zeros(0),
        nonlinear_candidates=(
            joint_dykstra_projector(C, np.zeros(2), cones=(soc_projector(0, [1, 2, 3]),)),
        ),
    )
    py, native = _pair(problem, np.zeros(n), max_iterations=600)
    assert py.iterations_used == native.iterations_used == 201


def test_far_dykstra_input_parity_reports_the_same_cap():
    candidate = joint_dykstra_projector(
        np.array([[1.0, 0.0, 0.0]]),
        np.array([-0.5]),
        members=(ball_projector([0, 1, 2], 1.0),),
        max_iterations=100,
    )
    problem = OptimizationProblem(
        np.zeros((3, 3)),
        np.zeros(3),
        np.zeros((0, 3)),
        np.zeros(0),
        nonlinear_candidates=(candidate,),
    )
    py, native = _pair(
        problem,
        np.array([2.7e8, 7.8e5, 4.0]),
        max_iterations=1,
        convergence={"enable_early_stopping": False},
    )
    assert py.dykstra_inner_cap_hits == native.dykstra_inner_cap_hits == 1
    assert py.dykstra_inner_converged.tolist() == native.dykstra_inner_converged.tolist() == [False]


def test_scoped_far_dykstra_input_parity_reports_the_same_cap():
    candidate = DykstraProjector(
        (
            halfspace_projector([1.0], -0.5, coordinates=[0]),
            ball_projector([0, 1, 2], 1.0),
        ),
        coordinates=[0, 1, 2],
        max_iterations=100,
    )
    problem = OptimizationProblem(
        np.zeros((4, 4)),
        np.zeros(4),
        np.zeros((0, 4)),
        np.zeros(0),
        nonlinear_candidates=(candidate,),
    )
    py, native = _pair(
        problem,
        np.array([2.7e8, 7.8e5, 4.0, 1e12]),
        max_iterations=1,
        convergence={"enable_early_stopping": False},
    )
    assert py.dykstra_inner_cap_hits == native.dykstra_inner_cap_hits == 1
    assert py.dykstra_inner_converged.tolist() == native.dykstra_inner_converged.tolist() == [False]


@pytest.mark.parametrize(
    "candidate,n",
    [
        (CutterCandidate(lambda x: float(x[0] - 1.0), lambda x: np.array([1.0])), 1),
        (ProjectorCandidate(lambda x: np.asarray(x, dtype=float)), 1),
    ],
    ids=["callback-cutter", "opaque-projector"],
)
def test_unsupported_native_candidates_reject_with_fallback(candidate, n):
    problem = OptimizationProblem(
        np.eye(n), np.zeros(n), np.zeros((0, n)), np.zeros(0), nonlinear_candidates=(candidate,)
    )
    with pytest.raises(ValueError, match=r"candidate 0.*backend='python'"):
        SNNSolver(problem, SolverConfig(backend="c"))


def test_released_polyhedral_near_zero_plateau_keeps_old_gate():
    problem = OptimizationProblem(
        np.diag([1.0, 0.1]), np.zeros(2), np.array([[1.0, 1.0]]), np.array([-10.0])
    )
    results = []
    for backend in ("python", "c_serial"):
        cfg = SolverConfig(
            backend=backend,
            max_iterations=1200,
            record_trajectory=False,
            convergence=ConvergenceConfig(optimality_test="none"),
        )
        results.append(SNNSolver(problem, cfg).solve(np.ones(2)))
    assert [r.iterations_used for r in results] == [501, 501]
    assert all(r.converged for r in results)


def test_permuted_affine_coordinates_reject_before_native_dispatch():
    affine = AffineSubspaceProjector(np.array([[1.0, 2.0]]), np.array([0.4]), coordinates=[1, 0])
    problem = OptimizationProblem(
        np.zeros((2, 2)), np.zeros(2), np.zeros((0, 2)), np.zeros(0), nonlinear_candidates=(affine,)
    )
    with pytest.raises(ValueError, match=r"candidate 0.*backend='python'"):
        SNNSolver(problem, SolverConfig(backend="c_serial"))


def test_nested_dykstra_member_rejects_with_member_path():
    ball = ball_projector([0, 1], 1.0)
    inner = DykstraProjector((ball, ball), name="inner")
    outer = DykstraProjector((inner, halfspace_projector([0.0, 1.0])))
    problem = OptimizationProblem(
        np.eye(2), np.zeros(2), np.zeros((0, 2)), np.zeros(0), nonlinear_candidates=(outer,)
    )
    with pytest.raises(ValueError, match=r"candidate 0 member 0.*backend='python'"):
        SNNSolver(problem, SolverConfig(backend="c_serial"))


def test_scoped_dykstra_rejects_member_outside_block():
    scoped = DykstraProjector((ball_projector([1], 1.0),), coordinates=[0])
    problem = OptimizationProblem(
        np.eye(2), np.zeros(2), np.zeros((0, 2)), np.zeros(0), nonlinear_candidates=(scoped,)
    )
    with pytest.raises(ValueError, match=r"candidate 0 member 0.*backend='python'"):
        SNNSolver(problem, SolverConfig(backend="c_serial"))


def test_dykstra_member_events_do_not_consume_winner_round_cap():
    members = (
        ball_projector([0, 1], 1.0, center=[0.0, 0.0]),
        ball_projector([0, 1], 1.0, center=[1.2, 0.0]),
    )
    problem = OptimizationProblem(
        np.zeros((2, 2)),
        np.zeros(2),
        np.zeros((0, 2)),
        np.zeros(0),
        nonlinear_candidates=(DykstraProjector(members, tolerance=1e-14, max_iterations=10),),
    )
    for cap in (3, 4):
        results = []
        for backend in ("python", "c_serial"):
            cfg = SolverConfig(
                backend=backend,
                max_iterations=2,
                max_projection_iters=cap,
                observe_projection_events=True,
                convergence=ConvergenceConfig(enable_early_stopping=False, optimality_test="none"),
            )
            results.append(SNNSolver(problem, cfg).solve(np.array([3.0, 2.0])))
        assert [r.n_projections for r in results] == [4, 4]
        assert [r.projection_truncated_sweeps for r in results] == [0, 0]
        assert [r.projection_cap_rechecks for r in results] == [0, 0]


def test_dykstra_counts_two_inner_cap_hits_in_one_sweep():
    candidates = tuple(
        DykstraProjector(
            (ball_projector([i], 1.0), ball_projector([i], 1.0)),
            tolerance=1e-12,
            max_iterations=1,
            name=f"dykstra-{i}",
        )
        for i in range(2)
    )
    problem = OptimizationProblem(
        np.zeros((2, 2)),
        np.zeros(2),
        np.zeros((0, 2)),
        np.zeros(0),
        nonlinear_candidates=candidates,
    )
    for backend in ("python", "c_serial"):
        cfg = SolverConfig(
            backend=backend,
            max_iterations=1,
            continue_after_projection_budget=True,
            convergence=ConvergenceConfig(enable_early_stopping=False),
        )
        result = SNNSolver(problem, cfg).solve(np.array([2.0, 2.0]))
        assert result.dykstra_inner_cap_hits == 2
        assert result.n_projections == 4


def test_native_spike_stream_respects_history_switch():
    problem = OptimizationProblem(
        np.eye(3),
        np.zeros(3),
        np.zeros((0, 3)),
        np.zeros(0),
        nonlinear_candidates=(soc_projector(0, [1, 2]),),
    )
    for backend in ("python", "c_serial"):
        cfg = SolverConfig(
            backend=backend,
            max_iterations=8,
            record_spike_history=False,
            convergence=ConvergenceConfig(enable_early_stopping=False),
        )
        result = SNNSolver(problem, cfg).solve(np.array([-0.2, 1.4, -0.2]))
        assert result.spike_event_kinds == []
        assert result.spike_event_indices.size == 0
        assert result.nonlinear_event_counts["set"] > 0


def _raw_extended(candidate, x0):
    """Call the public native binding directly to exercise its range gate."""
    n = len(x0)
    problem = OptimizationProblem(
        np.zeros((n, n)),
        np.zeros(n),
        np.zeros((0, n)),
        np.zeros(0),
        nonlinear_candidates=(candidate,),
    )
    solver = SNNSolver(problem, SolverConfig(backend="c_serial"))
    top, members, coords, data = solver._native_descriptor_cache
    args = [
        np.zeros((n, n)),
        np.zeros(n),
        np.zeros((0, n)),
        np.zeros(0),
        np.zeros(0),
        np.zeros(0),
        np.zeros((0, 0)),
        np.asarray(x0, dtype=float),
        0.01,
        1e-8,
        1,
        100,
        False,
        0.0,
        False,
        0.0,
        top,
        coords,
        data,
        members,
        False,
        False,
    ]
    return args


@pytest.mark.parametrize("field,value", [(0, 99), (1, 999), (9, 999)])
def test_native_binding_rejects_bad_descriptor_fields(field, value):
    args = _raw_extended(ball_projector([0], 1.0), np.array([2.0]))
    top = args[16].copy()
    top[0, field] = value
    args[16] = top
    with pytest.raises(ValueError, match="candidate_meta row 0"):
        _kernel.solve_euler_extended(*args)


def test_native_binding_rejects_bad_coordinate_and_member_ranges():
    args = _raw_extended(ball_projector([0], 1.0), np.array([2.0]))
    args[17] = np.array([1], dtype=np.int64)
    with pytest.raises(ValueError, match="coordinate"):
        _kernel.solve_euler_extended(*args)

    dykstra = DykstraProjector((ball_projector([0], 1.0),))
    args = _raw_extended(dykstra, np.array([2.0]))
    top = args[16].copy()
    top[0, 5] = 999
    args[16] = top
    with pytest.raises(ValueError, match="invalid member range"):
        _kernel.solve_euler_extended(*args)


def test_native_dykstra_raises_on_nonfinite_correction():
    ball = ball_projector([0], 1.0)
    args = _raw_extended(DykstraProjector((ball, ball)), np.array([1e308]))
    with pytest.raises(ValueError, match="non-finite correction"):
        _kernel.solve_euler_extended(*args)


@pytest.mark.parametrize(
    "field,value",
    [
        (6, 9),  # rows above the native cap
        (2, 3),  # coordinate count does not match rows*cols
        (10, 0),  # spectral data count must be one
    ],
    ids=["spectral-cap", "spectral-coordinates", "spectral-data"],
)
def test_native_binding_rejects_spectral_fields(field, value):
    args = _raw_extended(spectral_ball_projector((2, 2)), np.array([2.0, 0.0, 0.0, 0.0]))
    top = args[16].copy()
    top[0, field] = value
    args[16] = top
    with pytest.raises(ValueError, match="candidate_meta row 0.*(spectral|coordinate)"):
        _kernel.solve_euler_extended(*args)


def test_native_binding_rejects_negative_spectral_radius():
    args = _raw_extended(spectral_ball_projector((2, 2)), np.array([2.0, 0.0, 0.0, 0.0]))
    data = args[18].copy()
    data[0] = -1.0
    args[18] = data
    with pytest.raises(ValueError, match="spectral"):
        _kernel.solve_euler_extended(*args)


def test_native_binding_rejects_psd_cap_and_member_cutter_kind():
    args = _raw_extended(psd_cone_projector(2), np.array([-1.0, 0.0, 0.0]))
    top = args[16].copy()
    top[0, 6] = 9
    args[16] = top
    with pytest.raises(ValueError, match="candidate_meta row 0.*PSD"):
        _kernel.solve_euler_extended(*args)

    args = _raw_extended(spectral_ball_projector((2, 2)), np.array([2.0, 0.0, 0.0, 0.0]))
    member = args[16][0].copy()
    member[0] = 7
    args[19] = np.asarray([member], dtype=np.int64)
    top = args[16].copy()
    top[0, 0] = 5
    top[0, 4] = 0
    top[0, 5] = 1
    top[0, 6] = 1
    top[0, 10] = 1
    args[16] = top
    with pytest.raises(ValueError, match="member_meta row 0 has unsupported kind"):
        _kernel.solve_euler_extended(*args)

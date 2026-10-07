"""Native parity and numerical coverage for spectral and PSD descriptors."""

from __future__ import annotations

import numpy as np
import pytest

from snn_opt import (
    ConvergenceConfig,
    CutterCandidate,
    DykstraProjector,
    OptimizationProblem,
    ProjectorCandidate,
    SNNSolver,
    SolverConfig,
    psd_cone_projector,
    spectral_ball_projector,
    spectral_norm_cutter,
)

_kernel = pytest.importorskip("snn_opt._kernel")


def _solve(candidate, x0, backend, *, iterations=1, n=None, k0=0.0):
    x0 = np.asarray(x0, dtype=float)
    n = x0.size if n is None else n
    p = OptimizationProblem(
        np.zeros((n, n)),
        np.zeros(n),
        np.zeros((0, n)),
        np.zeros(0),
        nonlinear_candidates=(candidate,),
    )
    cfg = SolverConfig(
        backend=backend,
        k0=k0,
        max_iterations=iterations,
        record_trajectory=True,
        convergence=ConvergenceConfig(enable_early_stopping=False),
    )
    return SNNSolver(p, cfg).solve(x0)


def test_jacobi_hooks_match_numpy_and_report_sweeps():
    rng = np.random.default_rng(478)
    worst_svd = 0
    for shape in ((3, 3), (4, 3), (3, 4), (8, 8)):
        a = rng.normal(size=shape)
        s, u, vt, sweeps = _kernel._test_jacobi_svd(a, 32, 4e-14)
        worst_svd = max(worst_svd, int(sweeps))
        np.testing.assert_allclose(s, np.linalg.svd(a, compute_uv=False), atol=2e-12)
        np.testing.assert_allclose(u @ np.diag(s) @ vt, a, atol=3e-12)
    worst_eigh = 0
    a = rng.normal(size=(8, 8))
    a = (a + a.T) / 2
    values, q, sweeps = _kernel._test_jacobi_eigh(a, 32, 4e-14)
    worst_eigh = int(sweeps)
    np.testing.assert_allclose(values, np.linalg.eigvalsh(a)[::-1], atol=3e-12)
    np.testing.assert_allclose(q @ np.diag(values) @ q.T, a, atol=3e-12)
    assert worst_svd <= 32 and worst_eigh <= 32


def test_jacobi_cap_exhaustion_is_fail_closed():
    with pytest.raises(ValueError, match="did not converge"):
        _kernel._test_jacobi_svd(np.array([[1.0, 2.0], [3.0, 4.0]]), 0, 1e-14)
    with pytest.raises(ValueError, match="did not converge"):
        _kernel._test_jacobi_eigh(np.array([[1.0, 2.0], [2.0, 1.0]]), 0, 1e-14)


@pytest.mark.parametrize("shape", [(3, 3), (4, 3), (3, 4), (8, 8)])
def test_spectral_projector_single_event_parity(shape):
    rng = np.random.default_rng(sum(shape))
    x = rng.normal(size=shape[0] * shape[1]) * 2.0
    candidate = spectral_ball_projector(shape, radius=0.7)
    py = _solve(candidate, x, "python")
    native = _solve(candidate, x, "c")
    np.testing.assert_allclose(native.final_x, py.final_x, atol=3e-10)
    assert native.spike_event_kinds == py.spike_event_kinds == ["set"]


def test_spectral_cutter_top_pair_parity_and_tie_result():
    x = np.diag([2.0, 0.8, 0.2]).reshape(-1)
    candidate = spectral_norm_cutter((3, 3), radius=1.0)
    py = _solve(candidate, x, "python")
    native = _solve(candidate, x, "c")
    np.testing.assert_allclose(native.final_x, py.final_x, atol=3e-10)
    tied = np.diag([2.0, 2.0, 0.0]).reshape(-1)
    py_t = _solve(candidate, tied, "python")
    native_t = _solve(candidate, tied, "c")
    np.testing.assert_allclose(native_t.final_x, py_t.final_x, atol=3e-10)


def test_psd_projector_embedded_and_dykstra_member():
    # packed order is (00, 01, 02, 11, 12, 22); coordinates are permuted.
    local = np.array([-1.0, 2.0, 0.0, 1.0, 0.0, 2.0])
    coords = [6, 1, 8, 3, 7, 5]
    x = np.zeros(10)
    x[coords] = local
    candidate = psd_cone_projector(3, coordinates=coords)
    py = _solve(candidate, x, "python", n=10)
    native = _solve(candidate, x, "c", n=10)
    np.testing.assert_allclose(native.final_x, py.final_x, atol=3e-10)
    dyk = DykstraProjector((psd_cone_projector(2),))
    y = np.array([-1.0, 0.5, 2.0])
    py = _solve(dyk, y, "python")
    native = _solve(dyk, y, "c")
    np.testing.assert_allclose(native.final_x, py.final_x, atol=3e-10)


def test_native_recognition_is_stamp_based_and_caps_shapes():
    spectral = spectral_ball_projector((9, 1), 1.0)
    p = OptimizationProblem(
        np.eye(9), np.zeros(9), np.zeros((0, 9)), np.zeros(0), nonlinear_candidates=(spectral,)
    )
    with pytest.raises(ValueError, match="8x8 cap"):
        SNNSolver(p, SolverConfig(backend="c"))
    psd = psd_cone_projector(9)
    p = OptimizationProblem(
        np.eye(45), np.zeros(45), np.zeros((0, 45)), np.zeros(0), nonlinear_candidates=(psd,)
    )
    with pytest.raises(ValueError, match="n=8 cap"):
        SNNSolver(p, SolverConfig(backend="c"))
    forged = ProjectorCandidate(spectral.project, kkt_data=dict(spectral.kkt_data))
    p = OptimizationProblem(
        np.eye(4), np.zeros(4), np.zeros((0, 4)), np.zeros(0), nonlinear_candidates=(forged,)
    )
    with pytest.raises(ValueError, match="backend='python'"):
        SNNSolver(p, SolverConfig(backend="c"))
    forged_cutter = CutterCandidate(
        lambda x: float(np.linalg.svd(x.reshape(2, 2), compute_uv=False)[0] - 1),
        lambda x: np.ones(4),
        kkt_data={"set": "spectral_ball", "shape": (2, 2), "radius": 1.0},
    )
    p = OptimizationProblem(
        np.eye(4), np.zeros(4), np.zeros((0, 4)), np.zeros(0), nonlinear_candidates=(forged_cutter,)
    )
    with pytest.raises(ValueError, match="backend='python'"):
        SNNSolver(p, SolverConfig(backend="c"))


def _batch_problem(blocks=16):
    rng = np.random.default_rng(478)
    hs = []
    for _ in range(blocks):
        h = rng.normal(size=(3, 3))
        hs.append(h / np.linalg.svd(h, compute_uv=False)[0])
    n = 9 * blocks
    candidates = tuple(
        spectral_norm_cutter(3, coordinates=range(9 * i, 9 * i + 9), name=f"block-{i}")
        for i in range(blocks)
    )
    return OptimizationProblem(
        0.2 * np.eye(n),
        -np.concatenate([h.ravel() for h in hs]),
        np.zeros((0, n)),
        np.zeros(0),
        nonlinear_candidates=candidates,
    ), hs


def test_native_16_block_batch_matches_python_and_closed_form():
    problem, hs = _batch_problem()
    results = {}
    for backend in ("python", "c"):
        cfg = SolverConfig(
            backend=backend,
            k0=2.5,
            max_iterations=40,
            record_trajectory=backend == "python",
            convergence=ConvergenceConfig(enable_early_stopping=False),
        )
        results[backend] = SNNSolver(problem, cfg).solve(np.zeros(144))
    np.testing.assert_allclose(results["c"].final_x, results["python"].final_x, atol=2e-8)
    native_blocks = results["c"].final_x.reshape(16, 3, 3)
    for i, h in enumerate(hs):
        closed = (
            np.linalg.svd(h, full_matrices=False)[0]
            * np.minimum(np.linalg.svd(h, compute_uv=False) / 0.2, 1.0)
        ) @ np.linalg.svd(h, full_matrices=False)[2]
        np.testing.assert_allclose(native_blocks[i], closed, atol=2e-7)


def test_chunked_early_stopping_cutter_parity():
    problem = OptimizationProblem(
        np.eye(4),
        np.zeros(4),
        np.zeros((0, 4)),
        np.zeros(0),
        nonlinear_candidates=(spectral_norm_cutter((2, 2), 0.8),),
    )
    results = {}
    for backend in ("python", "c"):
        cfg = SolverConfig(
            backend=backend, k0=0.2, max_iterations=30, record_trajectory=backend == "python"
        )
        results[backend] = SNNSolver(problem, cfg).solve(np.array([2.0, 0.0, 0.0, 1.0]))
    np.testing.assert_allclose(results["c"].final_x, results["python"].final_x, atol=2e-7)
    assert results["c"].spike_event_kinds == results["python"].spike_event_kinds

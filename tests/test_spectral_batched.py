"""Batched spectral-ball blocks (coordinates=): each block's event normal must be
local, and a batch of independent blocks must solve each block exactly
(regression for the ambient-copy jacobian leak, 2026-09-25)."""

import numpy as np

from snn_opt import (
    ConvergenceConfig,
    OptimizationProblem,
    SNNSolver,
    SolverConfig,
    spectral_norm_cutter,
)


def _closed_form(H, eps):
    U, s, Vt = np.linalg.svd(H)
    return (U * np.minimum(s / eps, 1.0)) @ Vt


def test_block_jacobian_is_zero_outside_its_coordinates():
    cutter = spectral_norm_cutter(3, coordinates=range(9, 18))
    x = np.arange(18, dtype=float) / 10.0
    g = np.asarray(cutter.jacobian(x), dtype=float).reshape(-1)
    if g.size == 18:
        assert np.all(g[:9] == 0.0)
        g = g[9:]
    assert g.size == 9
    M = x[9:18].reshape(3, 3)
    U, _, Vt = np.linalg.svd(M)
    assert np.allclose(np.abs(g), np.abs(np.outer(U[:, 0], Vt[0]).ravel()))


def _solve_batch(B, seed=3, eps=0.2, iters=40):
    rng = np.random.default_rng(seed)
    Hs = [rng.normal(size=(3, 3)) for _ in range(B)]
    Hs = [H / np.linalg.svd(H, compute_uv=False)[0] for H in Hs]
    n = 9 * B
    cands = tuple(spectral_norm_cutter(3, coordinates=range(9 * b, 9 * b + 9),
                                       name=f"blk{b}") for b in range(B))
    c = -np.concatenate([H.ravel() for H in Hs])
    prob = OptimizationProblem(eps * np.eye(n), c, np.zeros((0, n)), np.zeros(0),
                               nonlinear_candidates=cands)
    cfg = SolverConfig(k0=0.5 / eps, max_iterations=iters, constraint_tol=1e-10,
                       convergence=ConvergenceConfig(enable_early_stopping=False))
    res = SNNSolver(prob, cfg).solve(np.zeros(n))
    x = np.asarray(res.X[-1]).reshape(B, 3, 3)
    return res, max(np.linalg.norm(x[b] - _closed_form(Hs[b], eps))
                    for b in range(B))


def test_batch_of_16_blocks_matches_each_closed_form():
    res, err = _solve_batch(16)
    assert err < 1e-8, err


def test_large_batch_equals_independent_blocks_without_budget_abort():
    # 340 blocks exceeds the old cap threshold (1000 / (3 events x ... ) ~ 333
    # candidates without box facets); the batch must neither abort nor couple.
    B, iters = 340, 2
    res, _ = _solve_batch(B, iters=iters)
    assert getattr(res, "convergence_reason", "") != "projection_budget_exhausted"
    rng = np.random.default_rng(3)
    Hs = [rng.normal(size=(3, 3)) for _ in range(B)]
    Hs = [H / np.linalg.svd(H, compute_uv=False)[0] for H in Hs]
    xb = np.asarray(res.X[-1]).reshape(B, 3, 3)
    eps = 0.2
    cfg = SolverConfig(k0=0.5 / eps, max_iterations=iters, constraint_tol=1e-10,
                       convergence=ConvergenceConfig(enable_early_stopping=False))
    for b in (0, 1, 170, B - 1):
        prob = OptimizationProblem(eps * np.eye(9), -Hs[b].ravel(), np.zeros((0, 9)),
                                   np.zeros(0),
                                   nonlinear_candidates=(spectral_norm_cutter(3),))
        single = np.asarray(SNNSolver(prob, cfg).solve(np.zeros(9)).X[-1]).reshape(3, 3)
        assert np.linalg.norm(xb[b] - single) < 1e-10

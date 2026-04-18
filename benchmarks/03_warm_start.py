"""Figure 3 — warm-start speedup on a sequence of related QPs.

Builds a sequence of 30 small (n=5) QPs whose Hessian and constraint matrix
are *identical*, but whose linear cost ``b_k`` drifts slowly — a stylized
receding-horizon-control workload. Each problem is solved twice:

    1. cold-started from a fixed ``x = 0``,
    2. warm-started from the previous problem's solution.

Early stopping is enabled in both branches so the wall time and projection
count reflect *time-to-convergence*, not a fixed iteration cap. The
side-by-side plot shows that warm starting collapses both metrics — exactly
the property an SNN solver inherits from the underlying projected-gradient
flow, and the reason this dynamic suits MPC-style problems.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib.pyplot as plt

import figstyle  # noqa: F401
from snn_opt import OptimizationProblem, SNNSolver, SolverConfig, ConvergenceConfig


def build_sequence(n: int = 5, m: int = 4, n_problems: int = 30, seed: int = 1):
    rng = np.random.default_rng(seed)
    Q = rng.standard_normal((n, n))
    A = Q @ Q.T + 1.0 * np.eye(n)            # well-conditioned
    C = rng.standard_normal((m, n))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    d = -0.5 * np.ones(m)                    # all-zeros feasible (slack 0.5)

    base = rng.standard_normal(n)
    drift = rng.standard_normal(n) * 0.01    # very small per-step drift
    bs = [base + k * drift for k in range(n_problems)]
    return A, C, d, bs


def solve_one(A, b, C, d, x0):
    cfg = SolverConfig(
        max_iterations=4_000,
        convergence=ConvergenceConfig(
            enable_early_stopping=True,
            proj_grad_tol=1e-5,
            min_iterations=20,
            check_every=10,
            patience=2,
        ),
    )
    solver = SNNSolver(OptimizationProblem(A=A, b=b, C=C, d=d), cfg)
    t0 = time.perf_counter()
    res = solver.solve(x0)
    return res, time.perf_counter() - t0


def main() -> int:
    A, C, d, bs = build_sequence()
    n = A.shape[0]

    cold_t, cold_p, cold_i = [], [], []
    warm_t, warm_p, warm_i = [], [], []
    cold_x0 = np.zeros(n)
    warm_x = np.zeros(n)

    for b in bs:
        r_c, t_c = solve_one(A, b, C, d, cold_x0)
        cold_t.append(t_c)
        cold_p.append(r_c.n_projections)
        cold_i.append(r_c.iterations_used)

        r_w, t_w = solve_one(A, b, C, d, warm_x.copy())
        warm_t.append(t_w)
        warm_p.append(r_w.n_projections)
        warm_i.append(r_w.iterations_used)
        warm_x = r_w.final_x

    cold_t, warm_t = np.asarray(cold_t), np.asarray(warm_t)
    cold_i, warm_i = np.asarray(cold_i), np.asarray(warm_i)
    idx = np.arange(len(bs))

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.2))

    axes[0].plot(idx, cold_t * 1e3, marker="o", label="cold start", color=figstyle.PALETTE[0])
    axes[0].plot(idx, warm_t * 1e3, marker="s", label="warm start", color=figstyle.PALETTE[1])
    axes[0].set_xlabel("problem index $k$")
    axes[0].set_ylabel("wall time (ms)")
    axes[0].set_title("(a) per-problem solve time")
    axes[0].legend(frameon=False)

    axes[1].plot(idx, cold_i, marker="o", label="cold start", color=figstyle.PALETTE[0])
    axes[1].plot(idx, warm_i, marker="s", label="warm start", color=figstyle.PALETTE[1])
    axes[1].set_xlabel("problem index $k$")
    axes[1].set_ylabel("iterations to converge")
    axes[1].set_title("(b) per-problem iteration count")
    axes[1].legend(frameon=False)

    sp_t = cold_t.mean() / max(warm_t.mean(), 1e-9)
    sp_i = cold_i.mean() / max(warm_i.mean(), 1e-9)
    fig.suptitle(
        f"Warm-start vs cold-start on {len(bs)} drifting QPs (n={n}) — "
        f"mean speedup: {sp_t:.1f}× time, {sp_i:.1f}× iterations",
        y=1.04,
    )
    fig.tight_layout()

    paths = figstyle.save(fig, "03_warm_start")
    plt.close(fig)
    print("wrote:", *paths, sep="\n  ")
    print(
        f"summary — cold mean: {cold_t.mean()*1e3:.2f} ms / {cold_i.mean():.0f} iter; "
        f"warm mean: {warm_t.mean()*1e3:.2f} ms / {warm_i.mean():.0f} iter"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

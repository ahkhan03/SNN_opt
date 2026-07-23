"""Figure 3: warm-start speedup on a sequence of related QPs.

Builds a sequence of 30 small (n=5) QPs whose Hessian and constraint matrix are
*identical* but whose linear cost b_k drifts slowly, a stylised receding-horizon
control workload. Each problem is solved twice:

    1. cold-started from a fixed x = 0,
    2. warm-started from the previous problem's solution.

Early stopping is enabled in both branches, so both metrics reflect
time-to-convergence rather than a fixed iteration cap.

Iteration count is the headline because it is deterministic: rerun this script
and it reproduces exactly. Wall time is reported alongside it, as the median of
`REPEATS` timed runs per problem, because a single timing pass picks up
scheduler noise that has nothing to do with the solver (the previous version of
this figure showed two such spikes and they were indistinguishable from signal).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib.pyplot as plt

import figstyle
from snn_opt import ConvergenceConfig, OptimizationProblem, SNNSolver, SolverConfig

REPEATS = 5      # timed runs per problem; the median is plotted


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


def _config() -> SolverConfig:
    return SolverConfig(
        max_iterations=4_000,
        convergence=ConvergenceConfig(
            enable_early_stopping=True,
            proj_grad_tol=1e-5,
            min_iterations=20,
            check_every=10,
            patience=2,
        ),
    )


def solve_one(A, b, C, d, x0):
    """Solve once for the result, then time `REPEATS` more runs and take the median."""
    solver = SNNSolver(OptimizationProblem(A=A, b=b, C=C, d=d), _config())
    res = solver.solve(x0)

    times = []
    for _ in range(REPEATS):
        s = SNNSolver(OptimizationProblem(A=A, b=b, C=C, d=d), _config())
        t0 = time.perf_counter()
        s.solve(x0)
        times.append(time.perf_counter() - t0)
    return res, float(np.median(times))


def main() -> int:
    A, C, d, bs = build_sequence()
    n = A.shape[0]

    cold_t, cold_i, warm_t, warm_i = [], [], [], []
    cold_x0 = np.zeros(n)
    warm_x = np.zeros(n)

    for b in bs:
        r_c, t_c = solve_one(A, b, C, d, cold_x0)
        cold_t.append(t_c)
        cold_i.append(r_c.iterations_used)

        r_w, t_w = solve_one(A, b, C, d, warm_x.copy())
        warm_t.append(t_w)
        warm_i.append(r_w.iterations_used)
        warm_x = r_w.final_x

    cold_t, warm_t = np.asarray(cold_t) * 1e3, np.asarray(warm_t) * 1e3
    cold_i, warm_i = np.asarray(cold_i), np.asarray(warm_i)
    idx = np.arange(len(bs))

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.2))

    for ax, cold, warm, ylabel, title in (
        (axes[0], cold_i, warm_i, "iterations to converge", "(a) iterations, deterministic"),
        (axes[1], cold_t, warm_t, "wall time (ms)", f"(b) wall time, median of {REPEATS}"),
    ):
        ax.plot(idx, cold, marker="o", markersize=3.5, color=figstyle.BLUE, label="cold start")
        ax.plot(idx, warm, marker="s", markersize=3.5, color=figstyle.VERMILION, label="warm start")
        ax.set_xlabel("problem index $k$")
        ax.set_ylabel(ylabel)
        figstyle.panel_title(ax, title)
        ax.set_ylim(0, max(cold.max(), warm.max()) * 1.18)
        ax.legend(loc="lower right", ncol=2)

    # Warm start pays from the second problem onward; problem 0 has nothing to
    # reuse, so quote the speedup over the steady-state tail rather than the mean.
    sp_i = cold_i[1:].mean() / max(warm_i[1:].mean(), 1e-9)
    sp_t = cold_t[1:].mean() / max(warm_t[1:].mean(), 1e-9)
    fig.suptitle(
        f"Warm start vs cold start on {len(bs)} drifting QPs (n={n}): "
        f"{sp_i:.1f}x fewer iterations, {sp_t:.1f}x faster from k=1 onward",
        y=1.04,
    )
    fig.tight_layout()

    paths = figstyle.save(fig, "03_warm_start")
    plt.close(fig)
    print("wrote:", *paths, sep="\n  ")
    print(
        f"summary (k>=1): cold {cold_t[1:].mean():.2f} ms / {cold_i[1:].mean():.0f} iter; "
        f"warm {warm_t[1:].mean():.2f} ms / {warm_i[1:].mean():.0f} iter; "
        f"speedup {sp_i:.2f}x iterations, {sp_t:.2f}x time"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

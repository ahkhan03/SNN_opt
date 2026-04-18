"""Figure 1 — convergence diagnostics for a representative QP.

Solves a 50-D random PSD QP with linear inequalities and plots, side by side:

    (a) objective gap |f(x_t) - f*| vs iteration (log y),
    (b) iterate stability ||x_{t+1} - x_t||_2 vs iteration (log y) — the
        natural convergence metric for projected-gradient flows; for an
        inequality-constrained problem the raw gradient norm does *not* go
        to zero (a strictly active constraint balances it),
    (c) maximum constraint violation vs iteration (log y).

The reference optimum f* is taken from a high-iteration run so the gap is
well-defined; this is the same trick used in the SNN-X papers when a
closed-form optimum is not available.
"""

from __future__ import annotations

import sys
from pathlib import Path

# bootstrap so the script runs from a clean checkout
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib.pyplot as plt

import figstyle  # noqa: F401  (sets global rcParams on import)
from snn_opt import OptimizationProblem, SNNSolver, SolverConfig, ConvergenceConfig


def random_qp(n: int, m: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    Q = rng.standard_normal((n, n))
    A = Q @ Q.T + 0.1 * np.eye(n)             # PSD Hessian
    b = rng.standard_normal(n)
    C = rng.standard_normal((m, n))
    # Make the all-zeros point feasible (slack = 1 on every row).
    d = -np.ones(m) - np.maximum(C @ np.zeros(n), 0)
    x0 = rng.standard_normal(n) * 0.1
    return A, b, C, d, x0


def main() -> int:
    n, m = 50, 30
    A, b, C, d, x0 = random_qp(n, m, seed=7)

    # Reference optimum: long run, tight tolerances, no early stop.
    cfg_ref = SolverConfig(
        max_iterations=20_000,
        convergence=ConvergenceConfig(enable_early_stopping=False),
    )
    ref = SNNSolver(OptimizationProblem(A=A, b=b, C=C, d=d), cfg_ref).solve(x0)
    f_star = ref.final_objective

    cfg = SolverConfig(max_iterations=4_000)
    res = SNNSolver(OptimizationProblem(A=A, b=b, C=C, d=d), cfg).solve(x0)

    X = res.X
    iters = np.arange(len(X))
    gap = np.maximum(np.abs(res.objective_values - f_star), 1e-16)
    step = np.linalg.norm(np.diff(X, axis=0), axis=1)
    step = np.maximum(np.concatenate([[step[0]], step]), 1e-16)
    viol = np.maximum(res.constraint_violations, 1e-16)

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0))

    axes[0].semilogy(iters, gap, label=r"$|f(x_t) - f^\star|$")
    axes[0].set_xlabel("iteration $t$")
    axes[0].set_ylabel("objective gap")
    axes[0].set_title("(a) objective convergence")

    axes[1].semilogy(iters, step, color=figstyle.PALETTE[1])
    axes[1].set_xlabel("iteration $t$")
    axes[1].set_ylabel(r"$\|x_{t+1} - x_t\|_2$")
    axes[1].set_title("(b) iterate stability")

    axes[2].semilogy(iters, viol, color=figstyle.PALETTE[2])
    axes[2].set_xlabel("iteration $t$")
    axes[2].set_ylabel("max violation")
    axes[2].set_title("(c) feasibility")

    fig.suptitle(
        f"snn_opt convergence on a random {n}-D QP with {m} inequality constraints",
        y=1.04,
    )
    fig.tight_layout()

    paths = figstyle.save(fig, "01_convergence")
    plt.close(fig)
    print("wrote:", *paths, sep="\n  ")
    print(
        f"\nDiagnostic — converged={res.converged} ({res.convergence_reason}), "
        f"iter={res.iterations_used}, projections={res.n_projections}, "
        f"final gap={gap[-1]:.2e}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

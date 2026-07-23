"""Figure 1: convergence diagnostics for a representative QP.

Solves a 50-D random PSD QP with 30 linear inequalities and plots, side by side:

    (a) objective gap |f(x_t) - f*| against the EXACT optimum (log y),
    (b) iterate stability ||x_{t+1} - x_t||_2 (log y), the natural convergence
        metric for a projected-gradient flow: with a strictly active constraint
        balancing it, the raw gradient norm does not go to zero,
    (c) maximum constraint violation (log y).

The reference f* comes from `qpref.solve_exact`, an active-set KKT solve, and
NOT from a long run of `snn_opt` itself. That distinction matters: measuring the
solver against its own fixed point cannot reveal a standing offset between that
fixed point and the true minimiser, and on this problem there is one. The gap
descends geometrically for roughly 1800 iterations and then settles onto an
accuracy floor set by the step size k0. `04_accuracy_tuning.py` maps that floor
against k0; `docs/theory.md` explains where it comes from.
"""

from __future__ import annotations

import sys
from pathlib import Path

# bootstrap so the script runs from a clean checkout
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib.pyplot as plt

import figstyle
import qpref
from snn_opt import OptimizationProblem, SNNSolver, SolverConfig


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

    # Independent reference optimum (active-set KKT, exact to ~1e-10).
    x_star, f_star, active = qpref.solve_exact(A, b, C, d)

    cfg = SolverConfig(max_iterations=4_000)
    res = SNNSolver(OptimizationProblem(A=A, b=b, C=C, d=d), cfg).solve(x0)

    X = res.X
    iters = np.arange(len(X))
    gap = np.maximum(np.abs(res.objective_values - f_star), 1e-16)
    step = np.linalg.norm(np.diff(X, axis=0), axis=1)
    step = np.maximum(np.concatenate([[step[0]], step]), 1e-16)
    viol = np.maximum(res.constraint_violations, 1e-16)

    # Past the knee the iterate alternates between two points. Report both
    # branches of that cycle rather than a single meaningless average.
    tail = gap[int(0.75 * len(gap)) :]
    lo, hi = float(tail.min()), float(tail.max())

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.1))

    # (a) objective gap. Past the knee the iterate settles into a period-2 limit
    # cycle, so the raw trace alternates every step and plots as a solid block.
    # Splitting it by parity draws the two branches of the cycle instead.
    axes[0].semilogy(
        iters[::2], gap[::2], color=figstyle.OBJECTIVE, linewidth=1.2, label="even $t$"
    )
    axes[0].semilogy(
        iters[1::2], gap[1::2], color=figstyle.PURPLE, linewidth=1.2, label="odd $t$"
    )
    axes[0].set_xlabel("iteration $t$")
    axes[0].set_ylabel(r"$|f(x_t) - f^\star|$")
    figstyle.panel_title(axes[0], "(a) objective gap vs exact optimum")
    axes[0].legend(loc="upper right", title="limit cycle branch", title_fontsize=8.0)

    # (b) iterate stability.
    axes[1].semilogy(iters, step, color=figstyle.STABILITY, linewidth=1.1)
    axes[1].set_xlabel("iteration $t$")
    axes[1].set_ylabel(r"$\|x_{t+1} - x_t\|_2$")
    figstyle.panel_title(axes[1], "(b) iterate stability")

    # (c) feasibility.
    axes[2].semilogy(iters, viol, color=figstyle.FEASIBILITY, linewidth=1.1)
    axes[2].set_xlabel("iteration $t$")
    axes[2].set_ylabel("max constraint violation")
    figstyle.panel_title(axes[2], "(c) feasibility")

    fig.suptitle(
        f"snn_opt on a random {n}-D QP with {m} inequalities "
        f"({len(active)} active at the optimum)",
        y=1.03,
    )
    fig.tight_layout()

    paths = figstyle.save(fig, "01_convergence")
    plt.close(fig)
    print("wrote:", *paths, sep="\n  ")
    print(
        f"\nDiagnostic: converged={res.converged} ({res.convergence_reason}), "
        f"iter={res.iterations_used}, projections={res.n_projections}"
    )
    print(
        f"  exact f*            = {f_star:.12f}   (active set: {active.tolist()})\n"
        f"  snn_opt f           = {res.final_objective:.12f}\n"
        f"  limit cycle gap     = {lo:.3e} .. {hi:.3e} (period 2)\n"
        f"  ||x - x*||          = {np.linalg.norm(res.final_x - x_star):.3e}\n"
        f"  joint_feasible      = {res.joint_feasible}  "
        f"(max row distance {res.max_distance_rows:.2e})\n"
        f"  stationarity        = {res.stationarity_residual:.3e}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

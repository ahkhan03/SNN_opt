"""Figure 2 — projection-spike raster.

A 4-D quadratic ``min  (1/2)||x - c||^2`` is solved over a box-shaped polytope
``-1 <= x_i <= 1`` (encoded as eight inequality constraints). The target
vector ``c`` lies *outside* the box, so the constrained optimum sits at a
vertex of the polytope and the trajectory must press against several faces in
turn. This produces a visually informative projection-spike raster — the
picture that motivates the *SNN-as-an-optimizer* framing.

Each row in the upper panel is one constraint; each marker is a projection
event at that constraint's iteration, sized by the displacement norm. The
lower panel plots the objective gap on the same x-axis, so the relationship
between spike bursts and convergence is immediate.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib.pyplot as plt

import figstyle  # noqa: F401
from snn_opt import OptimizationProblem, SNNSolver, SolverConfig, ConvergenceConfig


def box_problem():
    """min ||x - c||^2 over the box [-1, 1]^4 with c outside the box.

    Encoded as eight inequalities:  x_i <= 1  and  -x_i <= 1.
    Constrained optimum is the vertex closest to c (clipped component-wise).
    """
    n = 4
    c = np.array([1.6, -1.4, 1.3, -1.7])    # outside the box
    A = 2.0 * np.eye(n)
    b = -2.0 * c
    C = np.vstack([np.eye(n), -np.eye(n)])
    d = -np.ones(2 * n)                     # x_i <= 1, -x_i <= 1
    x0 = np.zeros(n)                        # feasible interior start
    return A, b, C, d, x0


def main() -> int:
    A, b, C, d, x0 = box_problem()
    m = C.shape[0]

    cfg_ref = SolverConfig(
        max_iterations=5_000,
        max_projection_iters=20,
        convergence=ConvergenceConfig(enable_early_stopping=False),
    )
    f_star = SNNSolver(OptimizationProblem(A=A, b=b, C=C, d=d), cfg_ref).solve(x0).final_objective

    cfg = SolverConfig(max_iterations=400, max_projection_iters=20)
    res = SNNSolver(OptimizationProblem(A=A, b=b, C=C, d=d), cfg).solve(x0)

    iters = np.arange(len(res.objective_values))
    gap = np.maximum(np.abs(res.objective_values - f_star), 1e-16)

    fig, (ax_top, ax_bot) = plt.subplots(
        2,
        1,
        figsize=(8.5, 4.6),
        sharex=True,
        gridspec_kw={"height_ratios": [1.4, 1.0]},
    )

    constraint_labels = [f"$+x_{{{i+1}}}\\leq 1$" for i in range(4)] + [
        f"$-x_{{{i+1}}}\\leq 1$" for i in range(4)
    ]

    if len(res.spike_times) > 0:
        spike_iters = res.spike_times.astype(int)
        spike_norms = res.spike_norms
        max_norm = max(spike_norms.max(), 1e-12)
        sizes = 8.0 + 70.0 * (spike_norms / max_norm)
        for k, (it, active) in enumerate(zip(spike_iters, res.spike_constraints)):
            for j in active:
                ax_top.scatter(
                    [it],
                    [j],
                    s=[sizes[k]],
                    color=figstyle.PALETTE[j % len(figstyle.PALETTE)],
                    alpha=0.8,
                    edgecolors="white",
                    linewidths=0.3,
                )
    ax_top.set_yticks(np.arange(m))
    ax_top.set_yticklabels(constraint_labels)
    ax_top.set_ylabel("active constraint")
    ax_top.set_title("(a) projection-spike raster — dot size ∝ projection magnitude")
    ax_top.set_ylim(-0.5, m - 0.5)
    ax_top.invert_yaxis()
    ax_top.grid(axis="x", alpha=0.25, linestyle="--", linewidth=0.6)

    ax_bot.semilogy(iters, gap, color="#444")
    ax_bot.set_xlabel("iteration $t$")
    ax_bot.set_ylabel(r"$|f(x_t) - f^\star|$")
    ax_bot.set_title("(b) objective convergence on the same horizon")

    fig.suptitle(
        f"Projection-spike dynamics on a 4-D box-constrained QP "
        f"(m={m} cuts, {res.n_projections} projections)",
        y=1.02,
    )
    fig.tight_layout()

    paths = figstyle.save(fig, "02_spike_raster")
    plt.close(fig)
    print("wrote:", *paths, sep="\n  ")
    print(
        f"summary — projections={res.n_projections}, "
        f"unique spike events={len(res.spike_times)}, "
        f"final f={res.final_objective:.4e}, f*={f_star:.4e}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Figure 4: how close the network's fixed point sits to the true optimum.

The spiking dynamics converge to a fixed point of the discretised flow, not to
the exact minimiser of the QP. The two differ by an offset that shrinks with the
gradient step size k0 = k0_scale / L. On the benchmark problem from Figure 1 the
default k0_scale = 0.5 leaves an objective gap around 7e-4, which is invisible
if the run is scored against a long run of the same solver and obvious the
moment it is scored against an exact reference.

This figure separates the two things that set the achievable accuracy:

    (a) the fixed-point offset, which shrinks as k0 shrinks, and
    (b) the iteration budget, since a smaller k0 also means smaller steps and so
        more iterations are needed to reach that offset in the first place.

Sweeping k0 at several budgets shows both at once. At a generous budget the gap
keeps falling as k0 shrinks; at a tight budget the curve turns back up, because
the solver runs out of iterations before it arrives. The useful setting is the
knee, and where the knee sits depends on the budget you actually have.

The runs here use `record_trajectory=False`. Only the final point is needed, and
the lean path avoids accumulating a per-projection spike history that is a real
memory hazard at these iteration counts.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib.pyplot as plt

import figstyle
import qpref
from snn_opt import ConvergenceConfig, OptimizationProblem, SNNSolver, SolverConfig

# Reuse Figure 1's problem so the two figures describe the same solver run.
from importlib import import_module

random_qp = import_module("01_convergence").random_qp

K0_SCALES = (0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005)
BUDGETS = (5_000, 20_000, 80_000)


def main() -> int:
    A, b, C, d, x0 = random_qp(50, 30, seed=7)
    _, f_star, _ = qpref.solve_exact(A, b, C, d)
    problem = OptimizationProblem(A=A, b=b, C=C, d=d)

    gaps = np.zeros((len(BUDGETS), len(K0_SCALES)))
    projections = np.zeros_like(gaps)
    for i, budget in enumerate(BUDGETS):
        for j, ks in enumerate(K0_SCALES):
            cfg = SolverConfig(
                max_iterations=budget,
                k0_scale=ks,
                record_trajectory=False,
                convergence=ConvergenceConfig(enable_early_stopping=False),
            )
            res = SNNSolver(problem, cfg).solve(x0)
            gaps[i, j] = abs(res.final_objective - f_star)
            projections[i, j] = res.n_projections
            print(
                f"  budget={budget:>6}  k0_scale={ks:<6} "
                f"gap={gaps[i, j]:.3e}  projections={res.n_projections:>8}"
            )

    fig, ax = plt.subplots(figsize=(7.2, 3.6))

    colors = (figstyle.BLUE, figstyle.VERMILION, figstyle.GREEN)
    for i, budget in enumerate(BUDGETS):
        ax.loglog(
            K0_SCALES,
            gaps[i],
            marker="o",
            markersize=4.5,
            color=colors[i],
            label=f"{budget // 1000}k iterations",
        )
        # Direct-label each curve's knee: the best k0 at that budget.
        j = int(np.argmin(gaps[i]))
        ax.annotate(
            f"{gaps[i, j]:.0e}",
            xy=(K0_SCALES[j], gaps[i, j]),
            xytext=(0, -13),
            textcoords="offset points",
            ha="center",
            fontsize=7.5,
            color=colors[i],
        )

    # Mark the default so a reader can locate the shipped behaviour at a glance.
    ax.axvline(0.5, color=figstyle.MUTED, linestyle="--", linewidth=0.9, zorder=1)
    ax.annotate(
        "default 0.5",
        xy=(0.5, gaps.max()),
        xytext=(-5, -2),
        textcoords="offset points",
        ha="right",
        va="top",
        fontsize=8.0,
        color=figstyle.MUTED,
    )

    ax.set_xlabel(r"$k_{0}$ scale  (step size $k_0 = k_{0}\mathrm{scale}\,/\,L$)")
    ax.set_ylabel(r"$|f(x) - f^\star|$")
    ax.legend(
        loc="upper right",
        title="iteration budget",
        title_fontsize=8.0,
        bbox_to_anchor=(0.995, 0.62),
    )
    ax.set_title(
        "Accuracy of the fixed point vs step size, at three iteration budgets\n"
        "50-D QP with 30 inequalities (the Figure 1 problem)",
        loc="left",
        fontsize=10.0,
        pad=8,
    )
    fig.tight_layout()

    paths = figstyle.save(fig, "04_accuracy_tuning")
    plt.close(fig)
    best = np.unravel_index(np.argmin(gaps), gaps.shape)
    print("\nwrote:", *paths, sep="\n  ")
    print(
        f"\nbest gap {gaps[best]:.3e} at k0_scale={K0_SCALES[best[1]]}, "
        f"budget={BUDGETS[best[0]]}; default k0_scale=0.5 gives {gaps[-1, 0]:.3e}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

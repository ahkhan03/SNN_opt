"""Figure 2: projection-spike raster.

An 8-D quadratic is solved over a 16-facet random polytope whose unconstrained
minimiser lies well outside the feasible set, so the trajectory has to travel a
long way along the boundary before it settles. Each marker in the upper panel
is one projection event: a constraint became active and the state was pushed
back onto its face. The lower panel plots the objective gap on the same axis.

This problem is chosen over a box-constrained one deliberately. On a box the
answer is reached in about three iterations and every active face then fires on
every step, which produces a raster of solid bars with no structure to read.
Here the network instead *searches* for the active set:

* row 11 fires a short burst around t = 4..10 and then falls silent, a facet
  the trajectory brushes against on its way past;
* rows 5, 10 and 3 are recruited at t ~= 21, 23 and 41 and then fire on every
  subsequent step.

The rows that keep firing are exactly the active set of the true optimum, which
`qpref.solve_exact` computes independently. Reading that off the raster is the
practical payoff of the spiking view: the population's steady-state firing
pattern IS the active set, and the transients are the search that found it.
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
from snn_opt import OptimizationProblem, SNNSolver, SolverConfig

# Iterations shown. The run goes to 400; past ~60 the raster is unchanging,
# so a full-width axis would spend 85% of the figure on a static pattern.
WINDOW = 130


def polytope_problem(n: int = 8, m: int = 16, seed: int = 3, offset: float = 2.0):
    """QP over a random polytope with the unconstrained minimiser far outside.

    Rows of ``C`` are unit-normalised so that a violation is a true Euclidean
    distance to the face, which is also what v0.5's winner selection assumes.
    """
    rng = np.random.default_rng(seed)
    Q = rng.standard_normal((n, n))
    A = Q @ Q.T + 0.5 * np.eye(n)
    C = rng.standard_normal((m, n))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    d = -np.ones(m)                      # origin is interior, slack 1 on every row
    b = -A @ (rng.standard_normal(n) * offset)   # min at a point outside the polytope
    x0 = np.zeros(n)
    return A, b, C, d, x0


def main() -> int:
    A, b, C, d, x0 = polytope_problem()
    m = C.shape[0]

    _, f_star, active = qpref.solve_exact(A, b, C, d)

    cfg = SolverConfig(max_iterations=400)
    res = SNNSolver(OptimizationProblem(A=A, b=b, C=C, d=d), cfg).solve(x0)

    iters = np.arange(len(res.objective_values))
    gap = np.maximum(np.abs(res.objective_values - f_star), 1e-16)

    # Flatten the spike record into (iteration, row, magnitude) triples.
    events: list[tuple[int, int, float]] = []
    for t, rows_fired, norm in zip(
        res.spike_times.astype(int), res.spike_constraints, res.spike_norms
    ):
        for j in np.atleast_1d(rows_fired):
            events.append((int(t), int(j), float(norm)))

    fired = sorted({j for _, j, _ in events})
    lane = {j: k for k, j in enumerate(fired)}      # pack rows that never fire out
    active_set = set(active.tolist())

    fig, (ax_top, ax_bot) = plt.subplots(
        2,
        1,
        figsize=(8.6, 4.0),
        sharex=True,
        gridspec_kw={"height_ratios": [1.35, 1.0]},
    )

    mags = np.array([e[2] for e in events])
    scale = mags.max() if mags.size else 1.0
    for keep, color, label in (
        (True, figstyle.BLUE, "persistent (active at the optimum)"),
        (False, figstyle.VERMILION, "transient (released again)"),
    ):
        sel = [e for e in events if (e[1] in active_set) is keep]
        if not sel:
            continue
        ax_top.scatter(
            [e[0] for e in sel],
            [lane[e[1]] for e in sel],
            s=[3.0 + 11.0 * (e[2] / scale) for e in sel],
            color=color,
            alpha=0.75,
            linewidths=0,
            label=label,
        )

    ax_top.set_yticks(range(len(fired)))
    ax_top.set_yticklabels([f"row {j}" for j in fired])
    ax_top.set_ylim(-0.6, len(fired) - 0.4)
    ax_top.invert_yaxis()
    ax_top.set_ylabel("constraint")
    figstyle.panel_title(
        ax_top, "(a) projection-spike raster, marker size proportional to displacement"
    )
    ax_top.grid(axis="y", visible=False)
    ax_top.legend(loc="lower right", markerscale=1.8)

    ax_bot.semilogy(iters, gap, color=figstyle.INK, linewidth=1.2)
    ax_bot.set_xlabel("iteration $t$")
    ax_bot.set_ylabel(r"$|f(x_t) - f^\star|$")
    figstyle.panel_title(ax_bot, "(b) objective gap vs exact optimum, same horizon")
    figstyle.annotate_floor(
        ax_bot,
        float(np.median(gap[-100:])),
        f"accuracy floor {np.median(gap[-100:]):.0e}",
        x=0.985,
        ha="right",
        va="bottom",
    )

    # Everything happens in the first ~60 iterations; the pattern is then
    # unchanged out to t = 400. Crop so the transients are not a sliver.
    ax_bot.set_xlim(-3, WINDOW)

    fig.suptitle(
        f"Projection-spike dynamics on an 8-D QP over a {m}-facet polytope "
        f"(first {WINDOW} of 400 iterations, {res.n_projections} projections total)",
        y=1.01,
    )
    fig.tight_layout()

    paths = figstyle.save(fig, "02_spike_raster")
    plt.close(fig)
    print("wrote:", *paths, sep="\n  ")
    print(
        f"\nprojections={res.n_projections}, spike events={len(res.spike_times)}, "
        f"rows that ever fire={fired}"
    )
    print(
        f"  exact active set = {active.tolist()}  "
        f"(persistent rows in the raster: "
        f"{sorted(j for j in fired if j in active_set)})\n"
        f"  final gap        = {gap[-1]:.3e}   joint_feasible={res.joint_feasible}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

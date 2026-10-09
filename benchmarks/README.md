# Benchmarks

Each script in this directory runs a small experiment and writes a figure under
[`../figures/`](../figures/) in both PDF (for paper inclusion) and PNG (for
README and web embedding) formats. Figures share the visual system defined in
[`figstyle.py`](figstyle.py): a four-slot colour-blind-safe categorical palette
assigned in fixed order (drawn from Wong 2011, checked as a palette rather than
by eye), serif body text, hairline spines, and a recessive dotted grid. The same
module is imported by the example scripts under [`../examples/`](../examples/),
so every figure in the repository and on the website reads as one system.

## How to run

After `pip install -e ".[examples]"` from the repo root:

```bash
python benchmarks/run_all.py         # regenerate everything (~30 s)
python benchmarks/01_convergence.py  # one script at a time
```

Or invoke the scripts directly from a clean checkout (each one bootstraps
`src/` into `sys.path`).

## What each figure shows

| Script | Figure | What it demonstrates |
|---|---|---|
| [`01_convergence.py`](01_convergence.py) | [`figures/01_convergence.{pdf,png}`](../figures/) | Three-panel convergence diagnostic on a random 50-D QP with 30 inequalities: (a) objective gap against the exact optimum, split by parity because the iterate settles into a period-2 limit cycle, (b) iterate stability `‖x_{t+1}−x_t‖`, (c) maximum constraint violation. |
| [`02_spike_raster.py`](02_spike_raster.py) | [`figures/02_spike_raster.{pdf,png}`](../figures/) | An 8-D QP over a 16-facet polytope with the unconstrained minimiser far outside it. Top panel is the projection-spike raster; bottom panel is the objective gap on the same axis. Constraints are coloured by whether they are active at the true optimum, so the raster shows the network searching for the active set and then locking onto it. |
| [`03_warm_start.py`](03_warm_start.py) | [`figures/03_warm_start.{pdf,png}`](../figures/) | A sequence of 30 small drifting QPs (stylised MPC workload), solved once cold-started from `x = 0` and once warm-started from the previous solution. Iterations are the headline because they are deterministic; wall time is the median of five timed runs per problem. |
| [`04_accuracy_tuning.py`](04_accuracy_tuning.py) | [`figures/04_accuracy_tuning.{pdf,png}`](../figures/) | Objective gap against the exact optimum as `k0_scale` is swept, at three iteration budgets. Separates the fixed-point offset (falls with `k0`) from the budget needed to reach it (rises as `k0` falls), so each budget has a knee. |
| [`05_exact_projection.py`](05_exact_projection.py) | prints a table, no figure | The Figure 1 problem solved with the default greedy row sweep (objective gap stuck on the 6.7e-4 floor at 40k iterations) and with the rows passed as one exact `joint_projector` candidate (gap 2.1e-10 at the default certificate; 3.8e-10 from the optimum with `kkt_rel_tol=1e-9`). With an exact projection the projected-gradient fixed point is the exact optimum, so the floor disappears; each step costs an inner Dykstra loop. A second table adds the box `[-0.3, 0.3]` as a `box_projector` Dykstra member (gap 1.8e-11, certified at 2651 iterations, against 2.3e-3 for the greedy sweep with scalar bounds) and times it per backend: the native member runs about 20x faster than the Python reference with identical Dykstra cycle counts, and writing the box as 2n halfspace rows instead costs about 6x more per iteration. |

## Conventions

- **Reference optima come from [`qpref.py`](qpref.py)**, an active-set KKT solve
  that is exact to about 1e-10 on these problems and needs only NumPy.

  This replaced the previous convention, which took the reference from a long
  run of `snn_opt` itself with early stopping disabled. That convention scores
  the solver against its own fixed point, and therefore **cannot see a standing
  offset between that fixed point and the true minimiser**. On the Figure 1
  problem there is one, worth 6.7e-4 with default settings, and under the old
  convention it registered as a gap of about 1e-6 instead. If you add a
  benchmark, use `qpref.solve_exact`, not a long self-run.

- **Numerical floors**: log-axis quantities are clipped at `1e-16` to keep the
  curves on the page. This is purely cosmetic.

- **Reproducibility**: every script uses a fixed `numpy.random.default_rng`
  seed. Figures 1, 2 and 4 are deterministic and re-run identically; Figure 3
  reports wall-clock time, so its timing panel varies slightly between machines
  while its iteration panel does not.

- **Typography**: titles, labels and captions punctuate with commas, colons or
  parentheses, matching the style used across the documentation.

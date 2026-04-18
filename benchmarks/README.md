# Benchmarks

Each script in this directory runs a small experiment and writes a figure
under [`../figures/`](../figures/) in both PDF (for paper inclusion) and PNG
(for README and web embedding) formats. Figures share an academic-style
matplotlib look defined in [`figstyle.py`](figstyle.py): muted, color-blind
friendly palette (Wong 2011), serif body text, no top/right spines, light
dashed gridlines.

## How to run

After `pip install -e ".[benchmarks]"` from the repo root:

```bash
python benchmarks/run_all.py        # regenerate everything
python benchmarks/01_convergence.py # one script at a time
```

Or invoke the scripts directly from a clean checkout (each one bootstraps
`src/` into `sys.path`).

## What each figure shows

| Script | Figure | What it demonstrates |
|---|---|---|
| [`01_convergence.py`](01_convergence.py) | [`figures/01_convergence.{pdf,png}`](../figures/) | Three-panel convergence diagnostic on a random 50-D QP with 30 inequalities: (a) objective gap on a log axis, (b) iterate stability `‖x_{t+1}−x_t‖`, (c) maximum constraint violation. |
| [`02_spike_raster.py`](02_spike_raster.py) | [`figures/02_spike_raster.{pdf,png}`](../figures/) | A 4-D quadratic over a box `[−1,1]^4` with the unconstrained optimum *outside* the box. Top panel is the projection-spike raster (one row per inequality, one dot per spike, sized by displacement); bottom panel is the objective gap on the same iteration axis. The picture motivates the *SNN-as-an-optimizer* framing. |
| [`03_warm_start.py`](03_warm_start.py) | [`figures/03_warm_start.{pdf,png}`](../figures/) | A sequence of 30 small drifting QPs (stylized MPC workload), solved once cold-started from `x = 0` and once warm-started from the previous problem's solution. Shows the order-of-magnitude reduction in both wall time and iteration count. |

## Conventions

- **Reference optima** are obtained by running the solver to a tight
  tolerance with early stopping disabled; objective gaps are then computed
  against that reference. This is the same convention used in the SNN-X
  paper series.
- **Numerical floors**: log-axis quantities are clipped at `1e-16` to keep
  the curves on the page; this is purely cosmetic.
- **Reproducibility**: every script uses a fixed `numpy.random.default_rng`
  seed, so re-running produces byte-identical outputs.

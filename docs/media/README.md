# Animations

The two animations in the top-level README are recordings of real solver
runs, not illustrations.

| File | Problem |
|---|---|
| `descent_landscape.gif` | 2-D strongly convex QP (condition number 4) inside a four-wall polygon, default settings |
| `friction_cone.gif` | 3-D QP whose force must stay in the friction cone `‖f_t‖ ≤ 0.5 f_n`, default settings |

**What is drawn, and where it comes from.** Each scene is one ordinary
`SNNSolver(...).solve(x0)` call on the released `snn-opt` package. For every
Euler step the export records the committed iterate `x_k` (`result.X[k]`),
the drift target `y_k = x_k - k0 ∇f(x_k)` computed with the solver's own
`k0`, and the constraints that spiked on that step (`result.spike_times`,
`result.spike_event_kinds`, `result.spike_event_indices`). The export
asserts that `y_k` plus the recorded spike corrections equals `x_{k+1}` to
machine precision, so a drawn overshoot and reset is exactly the step the
solver took. Distances in the lower plot are to an optimum computed
independently of `snn_opt`: an equality-constrained KKT solve on the active
rows for the polygon, and Newton's method on the cone's KKT system for the
friction cone. The marker is drawn on the objective surface for intuition,
but the dynamics are first order: it has no momentum.

**How they are made.** The scenes are exported, rendered and captured by
the companion site's repository
([ahkhan03/SNN_web](https://github.com/ahkhan03/SNN_web)):
`scripts/export_scenes.py` (solver runs to JSON), the three.js player in
`src/scripts/spike-player.ts`, `scripts/capture.mjs` (deterministic frame
capture in headless Chromium, with `--orbit 0` for the fixed-camera GIFs)
and `scripts/encode.sh` (ffmpeg). The same player runs interactively at
[snn.ahkhan.me](https://snn.ahkhan.me).

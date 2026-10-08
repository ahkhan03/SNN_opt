# Feasibility and certification: what changed and why

The README states what the result fields mean today. This page keeps the
longer account of the two releases that changed them, for anyone comparing
results across versions or reproducing an older paper. Field-by-field
reference: [`api.md`](api.md). The pages below describe the polyhedral
solver as those releases left it. Since v0.7.0, `joint_feasible` also covers
nonlinear candidates, and for strongly convex problems on exact Dykstra, PSD
or spectral projectors the certificate becomes a state-unit error bound
instead of the gradient-unit fit described here; see
[`theory.md`](theory.md#8-beyond-polytopes-nonlinear-and-conic-constraints)
and [`api.md`](api.md#certificate-on-the-nonlinear-path).

## v0.5.0: bounds inside one projection sweep

v0.5.0 is a **structural correctness release**, and the behaviour it fixes is
worth understanding before relying on results from an earlier version.

Before v0.5.0, bound constraints were enforced by a terminal clip applied after
the halfspace sweep, with nothing re-projecting behind it. Composing the two is
not a projection onto their intersection (the classical POCS failure), so on a
problem where a bound and an interacting row are simultaneously active, the
solver could stall at a point feasible for neither and report an objective that
*undercuts* the true optimum. Bounds are now implicit unit-normal facets inside
one unified projection sweep, and the terminal clip is gone.

Three result fields expose feasibility, optimality diagnostics, and projection
termination on any nontrivial problem:

```python
result.joint_feasible            # feasibility of rows AND bounds together
result.kkt_residual              # scale-invariant KKT certificate (v0.6.0)
result.projection_budget_exhausted
```

* **`joint_feasible`** is the honest feasibility flag. Pre-0.5 the convergence
  gate looked at rows only, so a box violation could not fail it.
* **`kkt_residual`** is the scale-invariant KKT certificate at the final
  point (see the next section); with the default settings, `converged=True`
  means exactly that this certificate passed, together with feasibility and
  the plateau criterion, at three consecutive checkpoints. The older
  `stationarity_residual` diagnostic is retained for one compatibility
  release but mixes units and can depend on constraint row order; prefer
  `kkt_residual`.
* **`projection_budget_exhausted`** reports that the sweep hit its watchdog.
  `max_projection_iters` is now a safety cap (default `None`, auto-sized), and
  hitting it **aborts** the solve rather than being reported as convergence.

`projection_method='fixed'` combined with bounds now raises, because the legacy
fixed-step path cannot enforce bounds correctly without the clip that was
removed.

## v0.6.0: a scale-invariant convergence certificate

Before v0.6.0, `converged` required an **absolute** projected-gradient norm
below `1e-6`. That test had two structural defects, found when an MPC user ran
QPs whose gradient scale is ~1e10: (a) rescaling the objective rescales every
gradient, so on large-scale problems the flag could never fire at any solution
quality; and (b) the projected-gradient heuristic removes each active facet's
gradient component independently, so at a constrained optimum with correlated
active normals it stalls at a cross-term residue and is structurally nonzero
even at the exact optimum. `converged=False` therefore said nothing about
solution quality; solves on perfectly solvable problems ran to their iteration
cap by construction.

The v0.6.0 criterion is a **KKT-cone certificate**: one nonnegative
least-squares fit of `-∇f(x)` onto the cone of all unit-normalized facet
normals, augmented with a complementarity row so slack facets cannot absorb
the gradient, accepted when

```
r_kkt  <=  kkt_abs_tol + kkt_rel_tol * max(‖A x‖, ‖b‖, ‖Nᵀμ‖)
```

Both sides carry gradient units, so while the relative term dominates the
threshold the decision is invariant under positive objective rescaling,
constraint row order, row duplication, and per-row scaling: the same problem
certifies identically at natural scale and at 1e10x. (The `kkt_abs_tol`
floor deliberately takes over at near-zero gradient scales, the intentional
fallback that lets a genuinely-zero problem terminate.) The fit runs host-side on every backend: the compiled kernel advances
the dynamics in checkpoint-sized chunks and the same Python policy evaluates
each checkpoint, so `converged` means one thing everywhere (the FPGA
reference is unchanged and reports fixed-horizon results, which the host can
certify with the same function). The cheap plateau/feasibility gates are
evaluated first, so the certificate's NNLS cost is confined to
near-termination checkpoints, and end-to-end overhead is negligible (see
`benchmarks/`).

Migration: results from v0.5 remain reproducible with
`ConvergenceConfig(optimality_test="legacy_projected_gradient")`, which
preserves the old test verbatim. `converged=False` runs from v0.5 can
legitimately become `converged=True` (or stop ~100x earlier) under the new
criterion; nothing about the dynamics changed, only the stopping decision.
The default `kkt_rel_tol = 1e-4` is calibrated to the O(k0) fixed-point floor
of the default step size: it certifies the quality the dynamics genuinely
reach, roughly 1e-3 relative solution error on well-conditioned problems. It
is a residual tolerance, not an error bound: on a nearly singular Hessian a
small residual can coexist with a larger solution displacement.

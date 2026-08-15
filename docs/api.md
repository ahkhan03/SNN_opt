# API Reference

Public symbols re-exported from `snn_opt`. All importable as

```python
from snn_opt import (
    OptimizationProblem,
    SolverConfig,
    ConvergenceConfig,
    SolverResult,
    SNNSolver,
    solve_qp,
)
```

## `solve_qp(A, b, C, d, x0, ...) -> SolverResult`

Convenience function that wraps `OptimizationProblem` + `SolverConfig` +
`SNNSolver.solve` for one-shot QPs.

| Argument | Type | Notes |
|---|---|---|
| `A` | `(n,n) array` | PSD Hessian (use `np.zeros((n,n))` for an LP). |
| `b` | `(n,) array` | Linear cost. |
| `C` | `(m,n) array` | Inequality matrix. |
| `d` | `(m,) array` | Inequality offset; constraints are `Cx + d ≤ 0`. |
| `x0` | `(n,) array` | Initial iterate (may be infeasible). |
| `k0` | `float` or `None` | Gradient step. `None` ⇒ auto from `λ_max(A)`. |
| `t_end` | `float` | Simulation horizon for `'ivp'` mode. |
| `max_iterations` | `int` | Cap for `'euler'` mode. |
| `integration_method` | `'euler'` (default) or `'ivp'` | |
| `projection_method` | `'adaptive'` (default) or `'fixed'` | Adaptive eliminates `k1`. |
| `k0_scale` | `float` | Conservatism factor on auto step. Default `0.5`. |
| `lower_bound`, `upper_bound` | `float` or `None` | Box bounds, enforced as implicit facets of the unified projection sweep (e.g. SVM dual). |
| `enable_early_stopping` | `bool` | Convergence-based termination, default on. |
| `record_trajectory` | `bool` | Keep the full iterate trajectory + spike events (default `True`). `False` runs the lean path; the compiled backends imply `False`. |
| `backend` | `str` | `'python'` (default), `'c'` (auto), `'c_serial'`, or `'c_openmp'`. See [`SolverConfig`](#solverconfig). |
| `verbose` | `bool` | Print solver progress. |

Returns: a [`SolverResult`](#solverresult).

## `OptimizationProblem`

Dataclass holding `A, b, C, d`. Methods:

- `objective(x)`: evaluate `½ xᵀAx + bᵀx`
- `gradient(x)`: `Ax + b`
- `constraint_values(x)`: `Cx + d`
- `is_feasible(x)`: boolean
- `max_violation(x)`: scalar

## `SolverConfig`

Solver hyper-parameters with sensible defaults. Most users only ever set
`max_iterations`, `lower_bound`, `upper_bound`, and `convergence`.

| Field | Default | Meaning |
|---|---|---|
| `k0` | `None` | Step size; `None` auto-computes from `λ_max(A)`. |
| `k0_scale` | `0.5` | Multiplier on the auto step (lower = safer). |
| `t_end` | `100.0` | IVP mode horizon. |
| `max_step` | `0.1` | IVP mode max ODE step. |
| `constraint_tol` | `1e-6` | Tolerance for "constraint violated". |
| `max_projection_iters` | `None` | Safety watchdog on the inner projection sweep; `None` auto-sizes it to `max(1000, 10 * (m + #box facets))`. Hitting it **aborts** the solve with `convergence_reason='projection_budget_exhausted'`; it is not routine truncation. |
| `integration_method` | `'euler'` | `'euler'` or `'ivp'`. |
| `max_iterations` | `2000` | Outer-iteration cap (Euler). |
| `projection_method` | `'adaptive'` | `'adaptive'` or `'fixed'`. |
| `k1` | `0.05` | Projection step (only used when `projection_method='fixed'`). |
| `lower_bound`, `upper_bound` | `None` | Box bounds (implicit facets of the projection sweep). |
| `record_trajectory` | `True` | Store the full iterate trajectory + per-spike events. `False` runs the lean solve (final state only); the compiled backends always run lean. |
| `backend` | `'python'` | Solve backend. `'python'` is the NumPy reference. The compiled pybind11 kernel (dense + `projection_method='adaptive'` only) comes in three numerically identical variants differing only in matvec threading: `'c'` (auto: OpenMP multicore when the wheel was built with it *and* the problem is large enough to amortize it, else single-thread), `'c_serial'` (forced single-thread), `'c_openmp'` (forced multicore; raises if the build lacks OpenMP). Only the matvec is parallel; the Euler recurrence + greedy projection are serial. Honours `OMP_NUM_THREADS`; `snn_opt._kernel.HAS_OPENMP` / `max_threads()` report the build's capability. |
| `transform` | `None` | Optional problem transform (the *transform axis*). `None` = canonical solve. A name (`'eigenbasis'`) or a `Transform` instance opts in; the problem is solved in transformed coordinates and mapped back. Composes with any backend; implies the lean result. See [Transforms](#transforms). |
| `record_spike_history` | `True` | Keep per-spike arrays (`spike_times`, `spike_deltas`, ...). `False` drops them to bound memory on large projection budgets. |
| `observe_projection_events` | `False` | Opt-in constant-memory observer of committed projection events; populates the observer fields below. Default off preserves v0.5 numerical and allocation behavior. |
| `convergence` | `ConvergenceConfig()` | See below. |

## `ConvergenceConfig`

Since v0.6.0 the authoritative optimality criterion is a **scale-invariant KKT
certificate**: one augmented nonnegative least-squares fit of the gradient onto
the cone of all unit-normalized facet normals (rows and box bounds), with a
complementarity row appended, accepted when

```
r_kkt <= kkt_abs_tol + kkt_rel_tol * max(||A x||, ||b||, ||N^T mu||)
```

Both residual components carry gradient units, so while the relative term
dominates the threshold the decision is invariant under positive objective
rescaling, constraint row order, row duplication, and per-row scaling (the
`kkt_abs_tol` floor deliberately takes over at near-zero gradient scales:
the intentional fallback that lets a genuinely-zero problem terminate). A
second practical limit: certification at tolerances below the facet family's
conditioning floor (~machine epsilon times the condition number of the
active normals) is limited by the accuracy of the least-squares fit itself,
which varies with the SciPy version and the dense/sparse code path; the
shipped default sits orders of magnitude above that floor for any reasonably
conditioned family. The cheap window criteria and the feasibility gate run first;
the NNLS only runs when they already pass, so its cost is confined to
near-termination checkpoints. On the compiled backends the kernel is driven in
checkpoint-sized chunks and this certificate is evaluated host-side, so every
backend shares one stopping-policy implementation.

| Field | Default | Meaning |
|---|---|---|
| `enable_early_stopping` | `True` | Master switch. |
| `optimality_test` | `"kkt"` | `"kkt"` (scale-invariant certificate), `"legacy_projected_gradient"` (pre-v0.6 absolute test), or `"none"` (cheap criteria only). |
| `kkt_abs_tol` | `1e-9` | Absolute floor of the certificate threshold (matters only near zero gradient scale). |
| `kkt_rel_tol` | `1e-4` | Relative certificate tolerance. Calibrated to the O(k0) fixed-point floor of the default dynamics; it is a KKT-residual tolerance, **not** a solution-error bound. |
| `obj_rel_tol` | `1e-8` | Relative-objective plateau over `window_size`. |
| `x_rel_tol` | `1e-8` | Relative iterate change. |
| `feasibility_tol` | `1e-2` | Maximum violation to count as converged. |
| `check_every` | `50` | Stride between convergence checks. |
| `min_iterations` | `100` | No early-stop before this. |
| `window_size` | `10` | Plateau-detection window. |
| `patience` | `3` | Consecutive passing checks needed. |
| `use_objective_plateau` | `True` | Enable plateau criterion. |
| `use_solution_stable` | `False` | Off by default, prone to false positives. |
| `require_feasibility` | `True` | Insist on feasibility for "converged". |

**Deprecated aliases** (one compatibility release): `use_projected_gradient`
and `proj_grad_tol` are constructor-only `InitVar` parameters: they are
consumed at construction, never stored, and therefore invisible to
`dataclasses.replace()` / `asdict()` round-trips of a resolved config. The
legacy criterion's tolerance lives in the regular field
`legacy_proj_grad_tol` (default `1e-6`). Supplying either
selects `optimality_test="legacy_projected_gradient"` (or `"none"` for
`use_projected_gradient=False`) with a `DeprecationWarning`; combining them
with explicit new-style settings raises `ValueError`. They are never silently
mapped onto the KKT tolerances, because the two quantities have different
semantics. The legacy criterion compares an absolute projected-gradient norm
against `proj_grad_tol`, which cannot fire on large-gradient-scale problems
and is structurally nonzero at constrained optima with correlated active
normals; it is retained verbatim for reproducing pre-v0.6 runs.

## `SNNSolver(problem, config=None)`

The full solver. Use this (rather than `solve_qp`) when you want to amortize
problem construction across many warm-started solves.

- `solver.solve(x0, verbose=False) -> SolverResult`: run the dynamics from
  `x0` and return diagnostics.

## `SolverResult`

Returned by `solve_qp` and `SNNSolver.solve`. Notable fields:

- `final_x`, `final_objective`, `final_proj_grad_norm`: solution and summary.
- `converged`, `convergence_reason`, `iterations_used`: termination info.
- `t`, `X`: full trajectory `(T,)` and `(T, n)`.
- `objective_values`, `constraint_violations`: `(T,)` per iteration.
- `n_projections`: total projection sub-iterations.
- `spike_times`, `spike_deltas`, `spike_norms`, `spike_constraints`,
  `spike_violation_values`: per-spike diagnostics, the raw material for
  the projection-spike raster (see [`02_spike_raster.py`](../benchmarks/02_spike_raster.py)).
- `total_projection_distance`: sum of spike norms.
- `summary()`: human-readable one-line-per-statistic string.

### Projection-event observer fields (v0.6.0, opt-in)

With `SolverConfig(observe_projection_events=True)`, the result additionally
carries a constant-memory record of every *committed* projection event (all
`None` when the observer is off):

| Field | Meaning |
|---|---|
| `explicit_row_event_counts` | Per-explicit-row committed event counts, `(m,)`. |
| `implicit_lower_event_counts`, `implicit_upper_event_counts` | Per-coordinate implicit-bound event counts, `(n,)`. |
| `explicit_row_events`, `implicit_lower_events`, `implicit_upper_events` | Totals of the three count arrays. |
| `projection_event_digest` | Canonical unsigned 64-bit digest of committed candidate IDs in event order (outer-iteration and within-sweep ordinal tokens included); the empty stream has the fixed offset-basis value. Chained across compiled-kernel chunks, so it matches monolithic runs exactly. |
| `projection_event_digest_algorithm` | Frozen digest identifier (`fnv1a64-word-v2`). |
| `observed_total_projection_distance` | Sum of Euclidean norms of all committed corrections. Unlike the legacy `total_projection_distance` it does not depend on retained spike history, so it is meaningful on lean solves. |
| `projection_first_candidate_id`, `projection_last_candidate_id` | First/last canonical candidate IDs (rows `j`, lower facets `m+i`, upper facets `m+n+i`); `None` for an empty stream. |
| `projection_cap_rechecks` | Inner sweeps that consumed the projection cap and performed a fresh joint-violation recheck. |

### KKT certificate fields (v0.6.0)

Every solve reports the scale-invariant certificate at the final point,
regardless of which `optimality_test` governed the flag. With the default
`optimality_test="kkt"`, `converged=True` **means** this certificate passed
(together with feasibility and the cheap criteria) at `patience` consecutive
checkpoints.

| Field | Meaning |
|---|---|
| `optimality_test` | Which criterion governed `converged`. |
| `kkt_residual` | `hypot(stationarity, complementarity)` from one augmented NNLS over all unit-normalized facets. Unique under multiplier non-uniqueness and invariant to row order and duplication; the dimensional value scales WITH the objective (use `kkt_residual / kkt_scale` as the invariant normalized defect). NaN when the fit failed (see `kkt_fit_status`). |
| `kkt_stationarity_residual` | `‖∇f(x) + Nᵀμ‖₂` component. |
| `kkt_complementarity_residual` | `|s|ᵀμ / max(1, ‖x‖)` component (gradient units). |
| `kkt_scale` | `max(‖A x‖, ‖b‖, ‖Nᵀμ‖)`, the relative-tolerance reference. |
| `kkt_tolerance` | `kkt_abs_tol + kkt_rel_tol * kkt_scale` in force at the final point. |
| `kkt_fit_status` | `"ok"`, `"non_finite"`, `"fit_failed"`, or `"too_large"` (dense facet family beyond the certificate's memory guard). Anything but `"ok"` fails the gate closed. |

Interpretation caveat: a small KKT residual does not bound the solution error
without a conditioning constant; on a nearly singular Hessian a large
displacement along a weak-curvature direction leaves the residual small. Use
`kkt_residual / kkt_scale` as the comparable cross-problem quantity.

### Correctness and diagnostic fields (v0.5.0)

These fields separate termination, joint feasibility, and the remaining
optimality defect. `joint_feasible` and `projection_budget_exhausted` are direct
checks.

| Field | Meaning |
|---|---|
| `joint_feasible` | Feasibility of the rows of `C` **and** the bounds together. Before v0.5.0 the convergence gate was rows-only, so a bound violation could not fail it. This is the flag to check. |
| `stationarity_residual` | LEGACY (pre-v0.6) eps-KKT diagnostic: the maximum of NNLS stationarity, complementarity, and primal defects on an eps-active set. Its three terms carry different units and its value can depend on constraint row order at rank-deficient active sets; retained for one compatibility release. Prefer `kkt_residual`. |
| `final_proj_grad_norm` | LEGACY heuristic: per-facet independent gradient projection. Structurally nonzero at constrained optima with correlated active normals; not an optimality measure. |
| `projection_budget_exhausted` | The inner sweep hit its `max_projection_iters` watchdog. The solve **aborts** with `convergence_reason='projection_budget_exhausted'` rather than reporting success from a knowingly infeasible point. |
| `max_violation_rows_raw`, `max_distance_rows`, `max_violation_box` | The components behind `joint_feasible`: raw row residual, row residual as a Euclidean distance (`residual / ‖c_j‖`), and the worst bound violation. |

Spike IDs in `spike_constraints` cover the implicit bound facets too, in a
frozen order: rows in input order, then lower facets `0..n-1`, then upper facets
`0..n-1`. So lower facet `i` is reported as `m + i` and upper facet `i` as
`m + n + i`.

See the README's [Accuracy and tuning](../README.md#accuracy-and-tuning) section
for how to interpret the residual and tune `k0_scale` with the iteration budget.

## Transforms

`snn_opt.transforms` is the **transform axis**: an explicit, backend-agnostic
rewrite of the problem that is solved in transformed coordinates and mapped back.
Transforms operate on the problem data (`A, b, C, d`), not the solve loop, so they
compose with every backend. Opt in via `SolverConfig(transform=...)`; the
canonical solver is the default.

```python
from snn_opt import solve_qp, EigenbasisTransform
solve_qp(A, b, C, d, x0, ...)                                   # canonical
# via SolverConfig:
cfg = SolverConfig(transform='eigenbasis')                     # by name
cfg = SolverConfig(transform=EigenbasisTransform())            # by instance
```

| Symbol | Notes |
|---|---|
| `Transform` | Base class. Subclass and implement `forward(problem, x0, config)` (and usually `check_applicable`). |
| `EigenbasisTransform` (`'eigenbasis'`) | Rotates a symmetric-PSD Hessian into its eigenbasis (`A = VΛVᵀ`), so the dominant `O(n²)` `A @ x` gradient step becomes an `O(n)` elementwise product `Λ ⊙ ỹ`; constraints rotate to `Ĉ = CV` with the Gram/row-norms invariant, so the projection is unchanged. Recovers `x = V ỹ`. Since v0.5.0 **box bounds are accepted**: they are not rotation-invariant, so they are materialized as explicit rotated unit-norm rows (`m` grows by up to `2n`), giving up the implicit `O(1)` facet advantage under a transform. Best on the compiled backends and larger `n`. |

## Versioning

`snn_opt` follows [SemVer](https://semver.org). The public API listed above
is the *commitment surface*: anything else (`snn_opt.solver._private_helper`,
internal config defaults that are not in `ConvergenceConfig` /
`SolverConfig`) may change between minor releases.

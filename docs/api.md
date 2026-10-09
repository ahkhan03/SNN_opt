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

The opt-in nonlinear and conic constraint family (v0.7.0) is also exported
from the top-level package; see [Nonlinear and conic constraints](#nonlinear-and-conic-constraints).

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
| `nonlinear_candidates` | sequence | Optional cutters and exact-set projectors, see [Nonlinear and conic constraints](#nonlinear-and-conic-constraints). Default: none. |
| `verbose` | `bool` | Print solver progress. |

Returns: a [`SolverResult`](#solverresult).

## `OptimizationProblem`

Dataclass holding `A, b, C, d` and, optionally,
`nonlinear_candidates` (a tuple of `CutterCandidate` / `ProjectorCandidate`
objects; empty by default, in which case the problem runs exactly as before
v0.7.0). Use `np.zeros((0, n))`, `np.zeros(0)` for `C`, `d` when every
constraint is a candidate. Methods:

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
| `max_projection_iters` | `None` | Safety watchdog on the inner projection sweep; `None` auto-sizes it to `max(1000, 10 * (m + #box facets + #nonlinear candidates))`. Hitting it **aborts** the solve with `convergence_reason='projection_budget_exhausted'` (on the nonlinear path, unless `continue_after_projection_budget=True`); it is not routine truncation. |
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
| `continue_after_projection_budget` | `False` | Nonlinear path only, experiment mode: a capped extended sweep returns its truncated point and the outer loop continues instead of aborting. The polyhedral path ignores it. |
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
| `kkt_fit_status` | `"ok"`, `"non_finite"`, `"fit_failed"`, `"too_large"` (dense facet family beyond the certificate's memory guard), or `"not_available"` (a nonlinear candidate supplies no certificate data). Anything but `"ok"` fails the gate closed. |

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
| `joint_feasible` | Feasibility of the rows of `C`, the bounds, and (since v0.7.0) any nonlinear candidates, together, within `feasibility_tol`. Before v0.5.0 the convergence gate was rows-only, so a bound violation could not fail it. This is the flag to check. |
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

## Nonlinear and conic constraints

*New in v0.7.0. Opt-in: a problem without `nonlinear_candidates` runs
exactly the v0.6 polyhedral solver.*

`snn_opt.nonlinear` extends the projection sweep beyond halfspaces. A
**candidate** is one more constraint that competes in the same
winner-take-all sweep as the rows of `C` and the box facets: at every event
of the sweep the most-violated candidate fires one spike, the spike applies that
candidate's own correction, and the sweep repeats until every constraint
holds. Two kinds exist.

| Kind | Describes | Correction applied by a spike |
|---|---|---|
| `CutterCandidate` | a differentiable convex inequality `g(x) <= 0` | a step to the supporting halfspace at the current point, `x <- x - g(x) / ‖∇g(x)‖² · ∇g(x)` (exact for affine `g`) |
| `ProjectorCandidate` | a closed convex set `K` with a known Euclidean projector | an exact reset `x <- P_K(x)` |

The winner is chosen by **distance**, so all candidates and rows are
comparable: rows score `(c_j x + d_j) / ‖c_j‖`, box facets their violation,
cutters `max(g, 0) / ‖∇g‖`, and projectors `‖P_K(x) - x‖`. Ties keep the
frozen order: rows, lower facets, upper facets, then candidates in input
order.

```python
import numpy as np
from snn_opt import OptimizationProblem, SNNSolver, SolverConfig, scaled_soc_projector

# A contact force f = (f_t1, f_t2, f_n) kept in the friction cone ||f_t|| <= 0.5 f_n.
cone = scaled_soc_projector(t_index=2, z_indices=[0, 1], mu=0.5, name="friction cone")
p = np.array([1.53, -0.39, 1.29])             # desired force, outside the cone
problem = OptimizationProblem(np.eye(3), -p, np.zeros((0, 3)), np.zeros(0),
                              nonlinear_candidates=(cone,))
result = SNNSolver(problem, SolverConfig()).solve(np.array([0.0, 0.0, 1.0]))
```

### Coordinates

Every callback receives and returns the **full** state vector. A candidate
may declare `coordinates` (a tuple of state indices); it then acts on those
entries only, may return just the local block, and the solver checks that
nothing outside the declaration moved. Built-in factories set this for you.

### Built-in sets

| Factory | Set | Notes |
|---|---|---|
| `halfspace_projector(c, d=0.0, coordinates=None)` | `c x + d <= 0` | Exact projector; the set-valued twin of a row. |
| `affine_cutter(c, d=0.0, coordinates=None)` | `c x + d <= 0` | Cutter with the same residual and normalisation as a row (identity fixture). |
| `ball_projector(indices, radius, center=None)` | `‖x_I - center‖ <= radius` | Radial projector on the coordinates `I`. |
| `soc_projector(t_index, z_indices)` | `‖z‖ <= t` | Second-order (Lorentz) cone; exact closed form including the apex. |
| `scaled_soc_projector(t_index, z_indices, mu)` | `‖z‖ <= mu · t` | Friction cones; `mu > 0`. |
| `psd_cone_projector(shape, coordinates=None)` (alias `psd_projector`) | symmetric `X ⪰ 0` | State holds `svec(X)`: upper triangle row by row, off-diagonals times `√2`, so Euclidean distance equals Frobenius distance. Projection clips negative eigenvalues. |
| `spectral_ball_projector(shape, radius=1.0, coordinates=None)` | `‖X‖₂ <= radius` | State holds `X` row-major (`shape` = `(rows, cols)` or an int for square). Exact SVD clip. |
| `spectral_norm_cutter(shape, radius=1.0, coordinates=None)` (alias `spectral_ball_cutter`) | `‖X‖₂ <= radius` | Cutter on the leading singular pair; carries the exact projector for the certificate, including tied singular values. |
| `AffineSubspaceProjector(B, h)` | `B x = h` | Equality projector. If the state is `(x, q)` with `len(q) = len(h)`, it instead projects onto the graph `q = B x + h`. Used by the SOC lifts. |

Custom sets use the base classes directly:
`CutterCandidate(value, jacobian, name=..., normal=None, coordinates=None)` and
`ProjectorCandidate(project, name=..., coordinates=None, normal=None)`.
`normal(x)` is optional certificate metadata (an outward unit normal, or
`None` away from the boundary). A custom projector with neither `normal` nor
`kkt_data={"euclidean_project": project}` still runs, but its certificate
reports `kkt_fit_status="not_available"` and fails closed, so the solve never
reports `converged=True` under the default test. Supplying
`euclidean_project` (the exact Euclidean projector, usually the same callable)
also makes the candidate eligible for the state-unit certificate below.

### Intersections: `dykstra_projector`

Projecting onto `K_1 ∩ ... ∩ K_r` one set at a time does **not** give the
projection onto the intersection, and when several sets are active at the
optimum, separate candidates make the sweep alternate between them. Wrap
interacting sets in one Dykstra candidate instead:

```python
from snn_opt import AffineSubspaceProjector, dykstra_projector

grasp_set = dykstra_projector([AffineSubspaceProjector(G, -w), *cones], name="grasp set")
problem = OptimizationProblem(np.eye(9), np.zeros(9), np.zeros((0, 9)), np.zeros(0),
                              nonlinear_candidates=(grasp_set,))
```

`dykstra_projector(members, tolerance=1e-12, max_iterations=10000,
coordinates=None, name="dykstra")` returns a `DykstraProjector`, which runs
Dykstra's algorithm from zero corrections on every call (results never
depend on earlier calls). Members are `ProjectorCandidate`s or plain
callables. `joint_dykstra_projector(C, d, members=(), ...)` (short alias
`joint_projector`) adds one exact halfspace member per row of `C x + d <= 0`,
so rows and cones are projected jointly. The inner loop stops when the
member and cycle residuals fall below `tolerance * max(1, ‖state‖)`. If it
hits `max_iterations` first, the projector returns its last iterate and
records `cap_hit`; the solver then aborts with
`convergence_reason='projection_budget_exhausted'` unless
`SolverConfig(continue_after_projection_budget=True)`.
[`examples/example8_friction_cone_grasp.py`](../examples/example8_friction_cone_grasp.py)
runs the same grasp both ways: wrapped, it certifies in 201 iterations,
within 4e-13 of a Newton-polished reference; as separate candidates, it
stops at the iteration cap with a relative KKT defect of 5e-2.

### Second-order-cone lifts

`lift_soc_l1(A, b, K, c, e, f)` and `lift_soc_l2(A, b, K, c, e, f)` turn the
QP `min ½xᵀAx + bᵀx` subject to `‖K x + c‖ <= eᵀx + f` into a lifted problem
over `(x, z[, t])`, where `z = K x + c` and, when `‖e‖ > 1e-14`, `t = eᵀx + f`
(a smaller `e` is treated as zero).

* `lift_soc_l1` keeps the coupling as a pair of opposed rows in `C` and adds
  the cone (or, when `e` is zero, the ball `‖z‖ <= f`) as a projector.
* `lift_soc_l2` replaces the coupling rows by one `AffineSubspaceProjector`
  on the graph.

Both return a `LiftedSOCResult(problem, coordinates, cone)`: the lifted
`OptimizationProblem`, a dict of index arrays (`"x"`, `"z"`, `"t"`,
`"dimension"`, `"t_variable"`), and the cone candidate. Recover the original
variables with `result.final_x[coordinates["x"]]`.

### Backends and restrictions

The nonlinear path runs on the Euler integrator with adaptive projection.

| | `backend='python'` | `backend='c'` (and `'c_serial'`, `'c_openmp'`) |
|---|---|---|
| Custom callbacks (`CutterCandidate`, `ProjectorCandidate`) | yes | no: rejected at construction with the candidate index |
| Built-in sets (halfspace, ball, SOC, scaled SOC, affine subspace) | yes | yes |
| PSD cone, spectral ball, spectral cutter | any size | blocks up to 8×8; the spectral cutter at top level only |
| `DykstraProjector` | yes, including nested | one level (members must be built-ins) |
| `record_trajectory=False` (lean result) | not supported | yes |
| `transform=...`, `integration_method='ivp'`, `projection_method='fixed'` | not supported | not supported |

Unsupported combinations raise `ValueError` naming the offending setting.
The compiled path is checked against the Python path by the parity tests in
`tests/test_c_backend_conic_parity.py` and
`tests/test_c_backend_spectral_psd_parity.py`.

### Certificate on the nonlinear path

`converged` still means the certificate passed, together with feasibility
and the plateau rule, at `patience` consecutive checkpoints. Which
certificate runs depends on the candidates.

**Gradient units (the default).** The NNLS fit of the polyhedral path is
extended by adding each candidate's normal cone to the facet normals: the
unit normal of a cutter or of a smooth boundary point, and the full polar
cone at nonsmooth points such as a cone apex or tied singular values. All
fields keep the meaning given under [KKT certificate fields](#kkt-certificate-fields-v060),
and the decision is invariant to objective scaling. Balls, second-order and
friction cones, halfspaces and affine subspaces passed as bare candidates
are certified this way.

**State units.** The certificate instead bounds the error in the state
when all of the following hold: the objective is strongly convex (modulus
`μ` above a roundoff floor); every candidate carries an exact certificate
projector, `kkt_data["euclidean_project"]`, and the candidates act on
disjoint coordinates; and every row and bound is slack by more than the
resulting residual. The built-ins that carry one are `DykstraProjector`
(so any set wrapped in `dykstra_projector`), `psd_cone_projector`,
`spectral_ball_projector` and `spectral_norm_cutter`. With
`T(x) = P(x - α∇f(x))` and `0 < α <= 1/L`, `T` is a contraction with factor
at most `1 - αμ`, so

```
‖x - x*‖ <= ‖x - T(x)‖ / (α μ).
```

On this path `kkt_stationarity_residual = ‖z - T(z)‖ / (αμ)`,
`kkt_complementarity_residual = ‖x - z‖` (the displacement to a Dykstra
witness `z`; zero for a direct projector), `kkt_residual` is their **sum**
(the bound), `kkt_scale = max(1, ‖x‖)` and does not follow the objective,
and `kkt_multipliers` is empty. A Dykstra bound does not include the inner
projection error, so with the default `tolerance = 1e-12` the true error
can exceed a reported bound that is smaller than that; a tolerance coarser
than about one percent of the state acceptance window is refused
(`kkt_fit_status="fit_failed"`) rather than trusted. Non-strongly-convex
objectives, and problems with active or nearly active rows or bounds, use
the gradient-unit fit.

### Result fields for candidates

| Field | Meaning |
|---|---|
| `spike_event_kinds`, `spike_event_indices` | Per spike: kind (`"row"`, `"lo"`, `"hi"`, `"cutter"`, `"set"`) and index within that kind. Prefer these to `spike_constraints` for candidates. |
| `spike_constraints` | Canonical IDs: rows `j`, lower facets `m + i`, upper facets `m + n + i`, candidate `q` at `m + 2n + q` (bound slots stay reserved even without bounds). |
| `nonlinear_event_counts` | Event counts by kind and by candidate name; Dykstra members appear as `dykstra:<q>:<member name>`. |
| `max_violation_nonlinear` | Largest normalised candidate violation at the final point. |
| `dykstra_inner_iterations`, `dykstra_inner_projection_events`, `dykstra_inner_converged` | Per projection sweep: Dykstra cycles, member projections, and whether the inner loop met its tolerance. |
| `dykstra_inner_cap_hits` | Number of inner calls that hit `max_iterations`. |
| `kkt_multipliers` | Nonnegative certificate coefficients, candidate normals included (gradient-unit path); empty on the state-unit path. |

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

## Reference solver: `snn_opt.reference`

An independent, certified reference optimum for the same problem class, for
measuring solution errors against (it does not use the spiking solver).
NumPy and SciPy only; import it explicitly:

```python
from snn_opt.reference import solve_reference, ReferenceNotVerified

ref = solve_reference(A, b, C, d)          # raises if it cannot certify the point
ref.x, ref.error_bound                     # point and a bound on ||x - x*||
ref.as_dict()                              # plain lists/floats/str, YAML-ready
ref = solve_reference(A, b, C, d, on_unverified="return")  # x is None on failure
```

`A` must be symmetric positive definite and only inequalities `C x + d <= 0`
are accepted (eliminate equalities first). The QP is turned into a
least-distance problem by a Cholesky factor of `A` and solved with one
Lawson-Hanson NNLS call, so linearly dependent or nearly parallel active rows
need no rank decision.

Every returned point carries a certificate. With `mu = lambda_min(A)`,
non-negative multipliers `lam`, `r = A x + b + C^T lam` and
`g = -lam^T (C x + d) >= 0`, a feasible `x` satisfies

```
‖x - x*‖ <= (‖r‖ + sqrt(‖r‖² + 4 μ g)) / (2 μ).
```

Derivation, in exact arithmetic: `f(x) >= f(x*) + μ/2 ‖x - x*‖²` because `x` is
feasible and `x*` optimal; `f(x*) >= L(x*) >= L(x) - ‖r‖ ‖x - x*‖ + μ/2 ‖x - x*‖²`
for the Lagrangian `L = f + lam^T (C · + d)`, because `lam^T (C x* + d) <= 0`;
adding the two with `L(x) = f(x) - g` gives the quadratic inequality above.
Both steps need `x` exactly feasible, and with nearly parallel rows the optimal
multipliers can be very large, so even a roundoff-sized violation can cost an
unbounded amount. The implementation therefore certifies only points that are
provably feasible, and evaluates the bound rigorously:

- `x` counts as feasible only if each computed `(C x + d)_i` is at most
  `-gamma_k (|C_i| |x| + |d_i|)`, a rigorous forward-error bound, so the exact
  value is `<= 0`;
- `‖r‖` and `g` are replaced by upper bounds built from the same kind of
  forward-error terms (`gamma_k (|A||x| + |b| + |C|^T lam)` for `r`), and
  `mu` by `lambda_min - 10 n eps lambda_max`;
- assumptions: IEEE double arithmetic with round-to-nearest and no
  underflow/overflow, `gamma_k = k u / (1 - k u)` with `u = eps/2`,
  `k = 2 (n + 2)` for the feasibility gate (an `n`-term dot product plus one
  addition needs only `gamma_{n+1}`, so this is a valid enclosure) and
  `k = 2 (n + m + 2)` for `r` and `g`, and a LAPACK symmetric eigensolver with backward error at
  most `10 n eps ‖A‖`.

`(A, b)` is divided by the power of two nearest `lambda_max(A)` before the
solve (exact, `x*` unchanged), so the result does not depend on the units of
the objective; `mu`, `objective`, `stationarity`, `complementarity` and
`multipliers` are reported in the units of the given problem. What remains
hard is dimensionless: a constrained optimum far from the unconstrained
minimiser in the `A`-norm relative to the row offsets, with small
`lambda_min(A)`. Such calls come back `"unverified"`, not wrong.

Under these assumptions `error_bound` bounds the exact distance from the
returned floating-point `x` to the exact minimiser of the given problem. The
least-distance point lands on its active rows to roundoff, on either side, so
when it is not provably feasible the problem is solved again with slightly
tightened offsets `d_i + t_i` (a few evaluation floors plus the observed
violation). The certificate is still taken against the original rows and pays
for the tightening through `g`. A feasible set too thin for that margin gives
`"unverified"`.

A point that is not provably feasible, or whose bound exceeds
`max_error_bound` (default `1e-3 * max(1, ||x||)`, a sanity ceiling rather than
an accuracy target), is not verified. Because the bound grows like
`sqrt(g / mu)`, it cannot certify below roughly `sqrt(eps * kappa(A)) * ||x||`
(of order 1e-6 on the benchmark problems), although the point is usually far
more accurate. Agreement with another solver below that level is two-solver
agreement, not a certified error; gate `error_bound` against the smallest error
you report.

| Field | Meaning |
|---|---|
| `x`, `objective` | The point and its objective; `None` unless verified. |
| `status` | `"verified"`, `"unverified"`, or `"infeasible"`: reported infeasible by an LP (HiGHS, default tolerances); not certified; no point is returned either way. |
| `error_bound` | Bound on `‖x − x*‖` in the units of `x`. |
| `mu`, `stationarity`, `complementarity` | `lambda_min(A)` (as computed), and the upper bounds on `‖r‖` and `g` that entered the bound. |
| `max_violation` | `max(0, max_i (C x + d)_i / ‖C_i‖)`. |
| `active_rows`, `multipliers` | Rows tight at `x`; one multiplier per row of `C`. |
| `route`, `version` | `"ldp-nnls"` and the `snn_opt` version, for provenance in stored results. |

Failures: malformed input (shapes, non-finite data, `A` not symmetric positive
definite) raises `ValueError`; an uncertified point or an LP-reported infeasible set raises
`ReferenceNotVerified` (a `RuntimeError` carrying the same diagnostic fields
and `.result` with `x=None`) unless `on_unverified="return"`. The legacy
`solve_exact(A, b, C, d) -> (x, f, active_rows)` wrapper is kept, also exposed
as `benchmarks/qpref.py`.

## Versioning

`snn_opt` follows [SemVer](https://semver.org). The public API listed above,
including `snn_opt.reference` (`solve_reference`, `ReferenceResult`,
`ReferenceNotVerified`, `solve_exact`, `objective`; since 0.8.0), is the
*commitment surface*: anything else (`snn_opt.solver._private_helper`,
internal config defaults that are not in `ConvergenceConfig` /
`SolverConfig`) may change between minor releases.

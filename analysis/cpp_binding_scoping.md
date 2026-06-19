# C/C++ Binding Scoping — SNN-QP Hot Loop

Scoping/feasibility analysis for compiling the SNN-QP hot loop to C/C++ so that
wall-time comparisons against CVXPY / SciPy / OSQP measure the *algorithm*
rather than the Python interpreter, and so the same kernel seeds the Vitis HLS
FPGA port.

Target file analysed: `src/snn_opt/solver.py` (branch `projection-neuromorphic-analysis`, 928 lines).

---

## 1. Verdict

Feasible, low-risk, and high-value. The hot loop is small (~80 lines of real
algorithm), already cleanly isolated, and uses only four numerical primitives
(matvec, axpy, ReLU/threshold, argmax). A faithful C++ port is ~300 lines.

It is worth doing for two compounding reasons:

1. **Fair benchmarking.** Today a wall-time comparison of `snn_opt` vs CVXPY or
   OSQP compares a Python interpreter loop against a compiled C solver. That is
   not a comparison of algorithms; it is a comparison of implementation
   substrates. A compiled SNN-QP backend removes that confound and lets the
   papers report honest wall-time numbers.
2. **FPGA groundwork.** The Vitis HLS flow (see `project_snn_qp_fpga_deployment`)
   consumes C++. A clean, STL-free, allocation-free C++ kernel written now *is*
   the HLS kernel later, minus the `#pragma HLS` directives. Doing the CPU
   binding first means the HLS port starts from validated, parity-tested code.

---

## 2. Anatomy of the hot loop

The benchmark-relevant solve path is `integration_method='euler'` +
`projection_method='adaptive'` (the `ivp` path uses `scipy.solve_ivp` and is
only used for continuous-time illustrations — out of scope for the kernel).

The hot loop is `_solve_euler` (solver.py:501-589). Per outer iteration:

```
Phase 1  gradient step
    g  = A @ x + b                      # O(n^2)  dense matvec
    x  = x - k0 * g                     # O(n)    axpy

Phase 2  adaptive projection  (_project_adaptive, solver.py:691-743)
    repeat up to max_projection_iters (100):
        r = C @ x + d                   # O(m*n)  dense matvec
        j = argmax(r)                   # O(m)
        if r[j] <= constraint_tol: break
        k1 = r[j] / c_norms_sq[j]       # scalar
        x  = x - k1 * C[j]              # O(n)    axpy on one constraint row

Phase 3  box clip                       # O(n)    elementwise min/max

Phase 4  bookkeeping  (see section 3)
```

Convergence is checked every `check_every` (default 50) iterations by
`_check_convergence` (solver.py:396-462): objective-plateau over a window,
projected-gradient norm (`_compute_projected_gradient_norm`, solver.py:334-394),
and a feasibility gate, combined with a patience counter.

Numerical primitives needed by the kernel: dense matvec, axpy, elementwise
min/max (ReLU/box), argmax, dot product, L2 norm. Nothing else. No matrix
factorisation, no inverse, no eigensolve inside the loop (`k0` is auto-computed
once at construction from `λ_max(A)` — that stays in Python).

---

## 3. Algorithm vs instrumentation — the split that matters

A large fraction of the current per-iteration cost is **instrumentation for the
paper figures, not the algorithm**. The C++ kernel should expose a "lean" mode
that drops all of it; the Python path keeps it for figure generation.

Instrumentation currently in the hot path:

- **`trajectory.append(x_current.copy())` every iteration** (solver.py:540) —
  O(n) copy + list growth; the full `(iters+1, n)` array is only needed for
  spike-raster / trajectory figures.
- **`obj_history` fed every iteration** (solver.py:543) — `problem.objective(x)`
  is `0.5 * xᵀAx + bᵀx`, a *second* O(n²) matvec of A per iteration, even
  though the objective window is only read every 50 iterations.
- **`spike_info` dicts** — one Python dict allocated per projection
  sub-iteration (solver.py:737-741), each holding three NumPy arrays.
- **`_build_result`** (solver.py:810-860) recomputes `objective_values` and
  `constraint_violations` over the *entire* stored trajectory — another
  O(iters·n²) + O(iters·m·n) pass, all of it pure post-hoc instrumentation.

Net effect for a dense QP: the algorithm needs **one** O(n²) matvec per
iteration (the gradient); the current code does effectively **two** in the loop
(gradient + objective-for-logging) plus a third full pass in `_build_result`.
Roughly half the dense-matvec work in a `solve()` call is instrumentation.

**Consequence for the port:** the C++ kernel's benchmark entry point should
return only `{x_final, iterations_used, n_projections, converged,
convergence_reason}` plus an optional fixed-size downsampled convergence curve.
Trajectory and spike metadata stay behind a separate, opt-in instrumented path
(or simply stay Python-only — the figures are not timed).

This split is also a free **pre-C++ win**: even staying in pure Python, adding a
`record_trajectory=False` lean path (reuse `A@x` between gradient and objective;
compute the objective only on check iterations; skip trajectory storage; skip
the `_build_result` recompute) roughly halves dense-QP solve time and gives a
fairer Python baseline immediately. Recommended as step 0 — it also pins down
exactly the lean kernel surface the C++ port must mirror.

---

## 4. Where Python overhead actually dominates

- **Small problems** (n, m from ~2 to ~100 — most of the SNN-X portfolio: SVM
  dual, linear regression, KRR, collaborative filtering, traffic, denoise). The
  BLAS matvec is tiny; thousands of iterations each carry ~15-25 interpreted
  bytecode-heavy operations (attribute lookups on dataclasses, method dispatch,
  dict construction, list `.append`). Here a C++ port is a **10-100×** wall-time
  win. This is the regime the papers live in.
- **The projection inner loop** is the single worst Python offender: a
  Python-level `while` with an `np.argmax` and a per-sub-iteration dict
  allocation. This is exactly the "event-triggered argmax-and-loop control flow"
  flagged in memory as the neuromorphic-impurity — and it is where the
  interpreter is slowest. Compiling it is the biggest single win.
- **Large problems** (n in the hundreds-plus). BLAS gemv starts to dominate and
  the per-iteration Python overhead amortises; the C++ win shrinks toward
  ~1.5-3×. Not the papers' main regime, but the projection loop and convergence
  machinery still benefit.

---

## 5. The apples-to-apples comparison — what "fair" requires

CVXPY is a modelling layer; it canonicalises the problem and dispatches to a
compiled backend (OSQP / ECOS / SCS / Clarabel). SciPy has no native QP — the
realistic SciPy baseline is `scipy.optimize.minimize(method='SLSQP')` or a
`quadprog`-style routine. To make a wall-time table defensible:

1. **Compile SNN-QP.** Without this, axis (b) below is meaningless. Everything
   else here is downstream of the binding.
2. **Common solution accuracy, not a common tolerance knob.** Each solver's
   tolerance parameter means something different. Drive *all* solvers to the
   same objective gap `|f(x) - f*|` and the same max constraint violation, where
   `f*` is a high-iteration reference optimum — exactly the trick already used
   in `benchmarks/01_convergence.py`. Then wall-time is comparable.
3. **Time the solve, not the setup.** Exclude CVXPY canonicalisation and
   problem-object construction from the timed region; report setup separately so
   the modelling-layer cost is visible but not conflated.
4. **Prefer OSQP-direct as the primary fair baseline.** OSQP solves the same
   inequality-constrained QP, is a first-order method (same algorithmic family
   as projected-gradient — the honest comparison), and the `osqp` Python package
   is a thin wrapper over C. Use CVXPY(+OSQP) as a *second* row to quantify the
   modelling-layer overhead, and SLSQP as the SciPy row the user asked for.
5. **Report three axes separately:**
   - (a) **iterations / convergence behaviour** — implementation-independent;
     valid even today.
   - (b) **wall time, compiled-vs-compiled** — needs the binding.
   - (c) **solution quality** — objective gap + violation at the common target.

   Today only (a) and (c) are honestly reportable. The binding unlocks (b),
   which is the row reviewers will actually scrutinise.

---

## 6. Binding approach — options and recommendation

| Option | C/C++ artifact reusable for HLS? | Build complexity | NumPy interop | Verdict |
|---|---|---|---|---|
| **pybind11** | Yes — kernel is plain C++ | Moderate (header-only dep + ext build) | Excellent | **Recommended** |
| Cython | No — output is Cython dialect | Low | Good | Kernel would be written twice |
| Numba `@njit` | No — no C++ artifact | None | n/a | Rejected: no HLS seed; can't JIT dict/list bookkeeping without a rewrite anyway |
| ctypes + plain C | Yes — very HLS-friendly | Low | Manual marshalling | Viable; loses pybind11's array/error ergonomics |

**Recommendation: pybind11.** Write the numerical core in a restricted C++
subset — plain pointers and fixed/caller-allocated buffers, no STL containers
or dynamic allocation in the inner loop, no exceptions, no recursion — and let
pybind11 handle only the NumPy ↔ buffer marshalling at the boundary. The core
translation unit is then directly ingestible by Vitis HLS; the binding shim is
the only part that gets discarded for the FPGA build.

Rejecting Numba is deliberate: it would give a fast CPU number but **no**
artifact for the FPGA track, so the kernel would still have to be written in
C++ later — two implementations to keep in numerical sync instead of one.

---

## 7. C++ kernel design

**Surface (one function):**

```cpp
struct SnnQpResult { int iterations_used, n_projections;
                     bool converged; int reason_code; };

SnnQpResult snn_qp_solve_euler(
    const double* A, const double* b,         // n x n row-major, n
    const double* C, const double* d,         // m x n row-major, m
    const double* c_norms_sq,                 // length m, precomputed in Python
    int n, int m,
    double k0, double constraint_tol,
    int max_iterations, int max_projection_iters,
    /* convergence */ bool early_stop, int check_every, int min_iterations,
    int window_size, int patience,
    double obj_rel_tol, double proj_grad_tol, double feasibility_tol,
    /* box */ bool has_lower, double lower, bool has_upper, double upper,
    const double* x0,
    double* x_out,                            // length n  (caller-allocated)
    double* conv_curve_out, int conv_curve_len /* optional, downsampled */);
```

`k0` and `c_norms_sq` are computed once in Python at construction
(`_compute_adaptive_k0` needs an eigensolve — leave it in Python/SciPy) and
passed in, so the kernel itself is factorisation-free and fully HLS-mappable.

**Matvec choice.** Write the matvec as explicit pragma-friendly loops rather
than calling Eigen/BLAS. Rationale: one codebase serves both the CPU benchmark
and the HLS kernel (Eigen is not HLS-synthesisable). With `-O3 -march=native`,
explicit gemv auto-vectorises to within ~1.5-2× of OpenBLAS — negligible for
the portfolio's problem sizes, where eliminating Python overhead is the
dominant effect. If a paper instance is ever large enough for that gap to
matter, add an Eigen-backed matvec behind a compile flag; the algorithm body is
unchanged.

**Fidelity requirements (these gate whether the comparison is valid):**

- **`argmax` tie-break** must return the *first* maximal index, matching NumPy.
- **Convergence machinery is a faithful port**, not an approximation —
  `check_every`, `min_iterations`, `window_size`, patience counter, the
  "require ALL enabled criteria" rule, and the feasibility gate. The papers
  report `iterations_used`; if the C++ version stops one check-interval early or
  late, iteration-count comparisons are muddied.
- Floating-point: explicit-loop matvec vs BLAS gemv differ in the last few
  ULPs. Expect `final_x` agreement to ~1e-10, not bit-exact. That is fine for
  the papers (their cross-solver agreement claims are at 1e-6…1e-10) but the
  validation harness must assert with a tolerance, not `==`.

**Scope boundaries for v1:**

- Dense `A`, `C` only. Most of the QP portfolio is dense (SVM Gram matrix,
  KRR, regression). Sparse/block-banded `A` (datacenter MPC) is a follow-up —
  add a CSR matvec later; the loop structure is identical.
- `euler` + `adaptive` projection only. `fixed` projection is ~10 extra lines
  (it is simpler — no argmax). The `ivp` path stays Python.
- Backend, not replacement. Per the FPGA workflow, `solver.py` stays the golden
  reference; the C++ kernel is selected via `solve_qp(..., backend="c")` or
  `SNNSolver(backend="c")`, with a parity harness enforcing agreement.

---

## 8. Validation strategy

A golden-parity harness is mandatory before any benchmark number is trusted:

- Run the Python solver and the C++ backend on the same battery: all of
  `examples/`, the `benchmarks/` random QPs, and representative instances from
  the paper portfolio (SVM dual, regression, KRR, CF).
- Assert `final_x` agreement to ≤1e-10 (L∞), `iterations_used` **exact**,
  `n_projections` **exact**, `converged` and `convergence_reason` identical.
- Wire it into `tests/` so the backend cannot silently drift from the golden
  reference — this is the same discipline the FPGA `FPGASolver` backend will
  need, so the harness is itself reusable groundwork.

---

## 9. Packaging note

`snn_opt` is a public pure-Python package (`pip install snn_opt` just works).
Adding a compiled extension must not break that. Keep the C backend **optional**:
either an extra (`snn_opt[fast]`), or build-if-compiler-present with graceful
fallback to the Python path. For the paper benchmarks this is moot — the
workstation builds it locally — but it keeps the public artifact clean. Add
`pybind11` to `[build-system].requires` and the extension under
`src/snn_opt/_kernel/`.

---

## 10. Effort estimate and phasing

| Phase | Work | Estimate | Gating |
|---|---|---|---|
| 0 | Lean Python path (`record_trajectory=False`, reuse `A@x`, no `_build_result` recompute) | ~0.5 day | none |
| 1 | C++ euler/adaptive kernel + pybind11 + build wiring | ~1.5-2 days | none |
| 2 | Golden-parity harness + `tests/` integration | ~0.5-1 day | Phase 1 |
| 3 | Benchmark harness: SNN-QP-C vs OSQP / CVXPY / SLSQP, common-accuracy protocol, table + figure | ~1-2 days | Phase 2 |
| 4 | HLS-ise the kernel (add `#pragma HLS`, fixed-point study) | FPGA track | hardware (Kria boards) |

Phases 0-3 are ~4-5 focused days and need no hardware. Phase 4 is the existing
FPGA deployment track and stays hardware-gated.

---

## 11. Groundwork for SNN-PCA and the projection variants

SNN-PCA is **not** in the `min ½xᵀAx+bᵀx s.t. Cx+d≤0` form — it is power
iteration with normalisation spikes — so it needs its own kernel. But this QP
binding lays real groundwork for it:

- the pybind11 build wiring, the optional-extension packaging, and the
  golden-parity harness are all reused verbatim;
- the shared primitives (matvec, axpy, ReLU, dot, norm) cover most of what a
  power-iteration kernel needs;
- the HLS-readiness discipline (no STL/alloc/exceptions in the inner loop)
  transfers directly.

Worth noting separately: the **single-population projection variants** (penalty,
Heun, heavy-ball, Nesterov, exp-Euler) — held on the `future-research` branch for
staged release, not shipped in the package — would be *easier* to port than the
canonical adaptive solver: they are pure forward-Euler with no event-triggered
argmax inner loop and no data-dependent control flow, which makes them both
trivial C++ and the most HLS-friendly targets. Each is its own kernel, though;
the current `_kernel` accelerates only the canonical adaptive solver. The planned
single-population accelerated-solvers methods paper would benefit from the same
compiled-backend infra at near-zero marginal cost.

---

## 12. Implementation status (Phases 0 and 1 complete)

Built on branch `native-cpp-backend`.

**Phase 0 (lean Python path).** `SolverConfig.record_trajectory` (default True
keeps the instrumented path bit-identical to before; False selects the new
`SNNSolver._solve_euler_lean`). The lean path drops trajectory and spike-event
storage and fuses the gradient and plateau-check matvecs into one `A @ x` per
iteration. Measured 1.7 to 2.2x faster than the instrumented path, with
bit-identical iterates and iteration counts.

**Phase 1 (C++ kernel).** `SolverConfig.backend='c'` dispatches to a pybind11
extension `snn_opt._kernel`. Pure kernel in `src/snn_opt/_native/snn_qp_core.hpp`
(STL/alloc-free inner loop, HLS-ready), pybind11 glue in `bindings.cpp`, built
by `setup.py` as an optional extension (`python setup.py build_ext --inplace`).

**Validation.** `tests/test_c_backend_parity.py` (12-case battery, wired into
pytest). C backend vs Python lean path: `final_x` agrees to ~1e-15, and
`iterations_used` / `n_projections` are exact on every case.

**Measured speedup, C backend vs Python lean path** (median 11x):

| problem size | speedup |
|---|---|
| n=10..50 | 13x to 67x |
| n=100 | ~4x |
| n=200..300 | 1.4x to 2.1x |
| box / well-conditioned | 3x to 31x |

Versus the original instrumented solver the end-to-end speedup is roughly 2x
(Phase 0) times 11x (Phase 1), about 20x median.

**Lesson worth recording.** A scalar matvec sum reduction does not vectorise
under `-O3` alone, because floating-point addition is not associative; the
first build was 3x *slower* than NumPy on large problems. Fix: `#pragma omp
simd reduction(+:acc)` on the matvec and dot loops, compiled with
`-fopenmp-simd` (SIMD pragmas only, no OpenMP runtime). The loop body stays a
clean dot product, so it remains a good HLS pipeline target. Also: setuptools
does not track header dependencies, so the extension declares
`depends=[snn_qp_core.hpp]` to avoid silently stale builds.

**Not yet done (follow-ups).** Phase 3 benchmark harness vs OSQP / CVXPY /
SciPy; sparse-matrix kernel; `fixed` projection in C; README/docs mention of
the new options; the Eigen-backed CPU matvec variant (deferred by design).

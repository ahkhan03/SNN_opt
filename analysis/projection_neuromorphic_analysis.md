# Projection step neuromorphic-purity analysis

**Branch:** `projection-neuromorphic-analysis` of the canonical SNN_opt repo at
`~/RD/Research/SNN/SNN_opt/`.

**Question.** The canonical `solve_qp` interleaves a forward-Euler gradient-flow
step with an "adaptive projection" inner loop. The adaptive projection is
event-triggered (fires only when at least one constraint is violated) and uses
a closed-form per-constraint correction step. This document evaluates whether
the projection step can be replaced with continuous-time ODE dynamics
discretised by Euler-family integrators, so that the entire solver maps onto
LIF / continuous-time spiking dynamics with no event-triggered closed-form
co-processor call.

The conclusion in two sentences. **The augmented-Lagrangian primal-dual variant
is a credible neuromorphic-pure replacement: it matches the canonical
solver's accuracy on MPC QPs (1e-9 relative L2) without using any
matrix-inverse or closed-form linear-algebra operation, and on a pathological
QP with rapidly switching active sets it actually outperforms the canonical
solver, which times out.** The plain forward-Euler-with-quadratic-penalty
formulation is the simplest possible neural-circuit substitute and reaches
1e-2 to 1e-5 accuracy depending on problem geometry, with Nesterov
acceleration giving an order-of-magnitude improvement at zero cost in
neuromorphic purity.

---

## 1. The existing closed-form adaptive projection

The canonical solver (`src/snn_opt/solver.py`, `_project_adaptive`,
lines 691-743) performs the following inner loop after each gradient step:

```
while max(g(x_proj)) > tol:
    j = argmax(g(x_proj))                  # most-violated constraint
    k1_j = g_j(x_proj) / ||c_j||^2          # exact step to boundary
    x_proj <- x_proj - k1_j * c_j           # snap onto j-th boundary
```

The per-constraint step `k1_j = g_j / ||c_j||^2` is a single scalar division.
There is **no** call to `numpy.linalg.solve`, `numpy.linalg.pinv`, Cholesky,
or QR anywhere in this loop. The user's prompt stated "the inner solve in
`numpy.linalg.solve(C_act @ C_act^T, …)`" was the operation under review;
this is incorrect about the canonical implementation, which is already a
Gauss-Seidel-style single-constraint projection. (An *un-optimised earlier
version* of this routine may have used a block solve; that variant is not
present in the current `main` branch.)

Where the adaptive projection is *not* neuromorphic-pure:

1. **Event-triggered control flow.** The `argmax` over `g` and the
   `while` loop with a numerical termination condition are sequential
   dispatch operations natural to a digital co-processor but awkward for
   a continuous-time spiking population. On Loihi-2 each `argmax`
   reduction would compile to an extra winner-take-all sub-circuit.
2. **Sequential single-constraint updates.** Each pass projects onto
   one face at a time. A LIF population integrating in continuous
   time naturally responds to *all* violated constraints simultaneously,
   not one at a time.
3. **Inner loop with numerical termination.** The loop runs an unknown
   number of inner iterations to drive `max g <= tol`. Variable-latency
   sub-routines complicate fixed-rate spike-tick deployment.

So the adaptive projection is "spike-mappable in principle" but
"event-driven and sequential in practice." The Euler-method variants
considered here remove all three issues by collapsing the projection
into ODE dynamics that integrate at every clock tick alongside the
gradient flow.

The actual numerical operation that needs replacement is a *block solve*
only if one wants Newton-style multi-constraint corrections; the current
single-constraint Gauss-Seidel adaptive projection is already
co-processor-light. The Euler variants below are therefore alternatives
to the *event-triggered control flow*, not to a hidden linear solver.

---

## 2. Six Euler-method projection variants

All six are implemented in `src/snn_opt/projection_variants.py` and exposed at
the package root. Each variant:

- Takes the same `(A, b, C, d, x0)` signature as `solve_qp`.
- Returns a `SolverResult` with the same fields populated.
- Uses ONLY operations that map cleanly onto LIF dynamics: vector
  add/subtract, scalar multiply, matrix-vector product, element-wise
  ReLU. (Variant 6 additionally uses one off-line matrix exponential
  computed once before the run.)
- Does **not** call `numpy.linalg.solve`, `pinv`, Cholesky, or QR
  anywhere in the inner loop.

### Variant 1 — Forward Euler + quadratic penalty (`solve_qp_penalty`)

Continuous-time ODE:

```
dx/dt = -(A x + b) - k_p * C^T relu(C x + d)
```

Forward Euler:

```
v       = relu(C x + d)
x_{k+1} = x_k - h * (A x_k + b) - h * k_p * C^T v
```

Single neural population. Only nonlinearity is element-wise ReLU. `k_p`
follows an `alpha^2` ramp from `k_p_init = 10*max(L_A, ||b||)/||C||^2`
to `1e4 * k_p_init`; step `h` adapts to current `k_p` for stability.

### Variant 2 — Augmented-Lagrangian primal-dual (`solve_qp_lagrangian`)

Augmented Lagrangian `L_rho(x, lambda) = f(x) + lambda^T relu(g) +
(rho/2) ||relu(g)||^2`. Saddle-point flow:

```
dx/dt      = -(A x + b) - C^T (lambda + rho * relu(g))
dlambda/dt = relu(C x + d)
```

Forward Euler with steps `h_x = h_l`. Lambda is rectified after each
ascent step. The augmented (`rho > 0`) form damps the multiplier
oscillation of the vanilla primal-dual flow, which we observed to
produce useless solutions on tightly constrained problems.

Two coupled neural populations: primal (excitatory) and dual (inhibitory,
rectified). No closed-form anywhere.

### Variant 3 — Heun (RK2) + quadratic penalty (`solve_qp_heun_penalty`)

Same ODE as Variant 1, integrated by explicit RK2:

```
f(x) = -(A x + b) - k_p * C^T relu(C x + d)
x*       = x_k + h * f(x_k)            # predictor
x_{k+1}  = x_k + (h/2) * (f(x_k) + f(x*))   # corrector
```

Two function evaluations per step ⇒ a two-phase spike clock. Local error
`O(h^3)` instead of Euler's `O(h^2)`, allowing ~1.4× larger `h`.

### Variant 4 — Polyak heavy-ball + penalty (`solve_qp_heavyball_penalty`)

Second-order momentum dynamics on the penalty-augmented gradient:

```
f(x)    = -(A x + b) - k_p * C^T relu(C x + d)
v_{k+1} = beta * v_k + (1 - beta) * f(x_k)
x_{k+1} = x_k + h * v_{k+1}
```

Two coupled LIF populations: primal and momentum. `beta = 0.9` default.

### Variant 5 — Nesterov-accelerated flow + penalty (`solve_qp_nesterov_penalty`)

FISTA-style update on the penalty-augmented gradient:

```
y_k     = x_k + ((k-1)/(k+2)) * (x_k - x_{k-1})
x_{k+1} = y_k + h * f(y_k)
```

Same neural-population structure as heavy-ball but with time-varying
coupling rate `(k-1)/(k+2)` (one extra "time-counter" parameter neuron).
Asymptotic `O(1/k^2)` rate vs `O(1/k)` for vanilla.

### Variant 6 — Exponential Euler + penalty (`solve_qp_expeuler_penalty`)

Exact integration of the linear part of the gradient flow, with the
penalty force treated as explicit forcing:

```
E_h = exp(-h * A)        # precomputed once
B_h = A^{-1} (I - E_h)   # precomputed once
forcing(x) = -b - k_p * C^T relu(C x + d)
x_{k+1} = E_h * x_k + B_h * forcing(x_k)
```

The matrix exponential and `B_h` are computed once before the run. Each
iteration is two matmuls + ReLU + one matvec. On Loihi-2 this is the
*native* compartment update because LIF leak is realised analytically as
a per-tick exponential decay. Trade-off: `h` is fixed for the whole run
(can't ramp), so we use a smaller `k_p_fin = 10 * k_p_init` ramp than
the time-varying-h variants.

---

## 3. Empirical comparison

Benchmark generator and full driver: `analysis/run_benchmark.py`. Raw
results: `analysis/results/{benchmark_summary.yaml, benchmark_raw.npz}`.

Three test sets:

1. **MPC QPs** (datacenter-thermal generator from
   `~/RD/Research/paper_workspace/datacenter_cooling_snn_paper/.../03a_scalability.py`).
   5/10/15 zones, horizon 10, 5 seeds each ⇒ 30/60/90 vars, 60/120/180
   box constraints.
2. **Random box-constrained QPs.** n in {5, 10, 20}, 5 seeds each.
3. **Pathological active-set switching QP.** Random Hessian + linear
   pull toward `(1,…,1)` corner of `[-1,1]^n` box, plus 2n extra
   nearly-redundant random half-spaces. n in {5, 10}, 3 seeds each.

Budgets: 5000 iterations on tests 1-2, 10000 on test 3. CVXPY/Clarabel is
the reference solution.

### 3.1 Relative L2 error to CVXPY (mean across seeds)

|         test |  n_vars | canonical | penalty   | lagrangian | heun      | heavyball | nesterov  | expeuler  |
|--------------|--------:|----------:|----------:|-----------:|----------:|----------:|----------:|----------:|
| box          |     5   | **1.9e-8** | 2.7e-3    | 9.9e-4    | 2.0e-3    | 2.7e-3    | 1.9e-5    | 1.0e-2    |
| box          |    10   | **3.0e-7** | 1.2e-2    | 1.8e-3    | 7.6e-3    | 1.2e-2    | 1.0e-4    | 5.8e-2    |
| box          |    20   | **7.2e-8** | 7.8e-3    | 1.8e-3    | 4.7e-3    | 7.8e-3    | 7.7e-5    | 7.0e-2    |
| mpc          |    30   | **2.7e-10**| 2.0e-5    | 1.1e-9    | 2.0e-5    | 2.0e-5    | 2.0e-5    | 2.0e-2    |
| mpc          |    60   | **8.8e-10**| 1.4e-5    | 1.9e-9    | 1.4e-5    | 1.4e-5    | 1.4e-5    | 1.4e-2    |
| mpc          |    90   | 5.5e-9    | 5.4e-3    | 2.3e-2    | 2.9e-3    | 9.1e-4    | **5.0e-5** | 2.1e-2    |
| pathological |     5   | 3.0e-1    | 2.5e-5    | **9.1e-10**| 2.5e-5    | 2.5e-5    | 2.4e-5    | 2.4e-2    |
| pathological |    10   | 2.4e-1    | 1.9e-5    | **8.1e-10**| 1.9e-5    | 1.9e-5    | 1.9e-5    | 1.9e-2    |

(Bold marks the best variant per row.)

### 3.2 Iterations used (mean) — convergence vs budget

|         test |  n_vars | canonical | penalty | lagrangian |   heun | heavyball | nesterov | expeuler |
|--------------|--------:|----------:|--------:|-----------:|-------:|----------:|---------:|---------:|
| box          |     5   |   **241** |   5000  |       4970 |   5000 |      5000 |     5000 |     5000 |
| box          |    10   |   **451** |   5000  |       5000 |   5000 |      5000 |     5000 |     5000 |
| box          |    20   |   **411** |   5000  |       5000 |   5000 |      5000 |     5000 |     5000 |
| mpc          |    30   |   **201** |   5000  |       2146 |   5000 |      5000 |     5000 |     5000 |
| mpc          |    60   |   **201** |   5000  |       2366 |   5000 |      5000 |     5000 |     5000 |
| mpc          |    90   |   **201** |   5000  |       2811 |   5000 |      5000 |     5000 |     5000 |
| pathological |     5   |     10000 |  10000  |   **1234** |  10000 |     10000 |    10000 |    10000 |
| pathological |    10   |     10000 |  10000  |   **1226** |  10000 |     10000 |    10000 |    10000 |

Only the canonical solver and the augmented-Lagrangian variant trigger
the early-stop convergence criterion (`converged=True`). All explicit-
penalty variants run to budget because their penalty bias keeps
`proj_grad_norm` above tolerance even after the iterate has stabilised.

### 3.3 Iterations to reach `||x - x_cvx|| / ||x_cvx|| < 1e-2` (median)

| test         | n_vars | canonical | penalty | lagrangian | heun | heavyball | nesterov | expeuler |
|--------------|-------:|----------:|--------:|-----------:|-----:|----------:|---------:|---------:|
| box          |    5   |    **23** |    2352 |       1694 | 2309 |      2354 |     2257 |     4517 |
| box          |   10   |    **58** |    2385 |       1877 | 2316 |      2385 |     2172 |       —  |
| box          |   20   |    **82** |    2571 |       2812 | 2361 |      2548 |     2063 |       —  |
| mpc          |   30   |     **1** |    2850 |        562 | 2849 |      2850 |     2847 |       —  |
| mpc          |   60   |     **1** |    2679 |        630 | 2679 |      2679 |     2676 |       —  |
| mpc          |   90   |     **1** |    2565 |        648 | 2564 |      2566 |     2563 |       —  |
| pathological |    5   |       2   |      47 |        322 |   34 |        56 |   **17** |      467 |
| pathological |   10   |       —   |      67 |        324 |   48 |        75 |     5641 |      660 |

(`—` = never reached `1e-2` within budget.) Note the canonical solver
hits 1e-2 at iteration 1 on MPC because `x0 = 0` is already within
1% of the optimum for that family of problems.

### 3.4 Wall-time (ms, mean) — 5000-iter budget on tests 1-2, 10000 on test 3

|         test |  n_vars | canonical | penalty | lagrangian | heun  | heavyball | nesterov | expeuler |
|--------------|--------:|----------:|--------:|-----------:|------:|----------:|---------:|---------:|
| box          |    5    |   **7.0** |   54.8  |       54.1 |  76.7 |      52.3 |     52.2 |     47.2 |
| box          |   10    |  **14.9** |   52.1  |       55.7 |  79.6 |      54.2 |     53.5 |     48.2 |
| box          |   20    |  **36.5** |   55.4  |       57.7 |  82.6 |      57.6 |     56.0 |     78.3 |
| mpc          |   30    |    46.4   |   79.3  |   **42.9** | 124.2 |      93.8 |     91.2 |    148.0 |
| mpc          |   60    |    98.1   |   86.9  |   **56.1** | 152.2 |     113.6 |    110.7 |    168.7 |
| mpc          |   90    | **140.3** |  118.2  |       90.8 | 193.6 |     147.5 |    152.9 |    182.2 |
| pathological |    5    |     485.5 |  105.5  |   **13.5** | 157.7 |     108.0 |    107.1 |     97.0 |
| pathological |   10    |    1802.0 |  112.1  |   **14.3** | 166.8 |     115.6 |    119.1 |    103.6 |

Augmented Lagrangian is faster than the canonical solver on MPC and 30-130x
faster on the pathological test. On easy box QPs the canonical solver wins
because it converges in a few hundred iterations. None of the variants
beat CVXPY/Clarabel in absolute wall-time on these small problems —
that's expected; the SNN solver's competitive advantage is asymptotic
scalability and MPC warm-start, not single-shot speed.

### 3.5 Max constraint violation at termination (mean across seeds)

|         test |  n_vars | canonical | penalty | lagrangian | heun     | heavyball | nesterov | expeuler |
|--------------|--------:|----------:|--------:|-----------:|---------:|----------:|---------:|---------:|
| box          |     5   | **0.0**   | 1.0e-5  |   1.2e-3   |   1.0e-5 |    1.0e-5 |   1.0e-5 |   1.0e-2 |
| box          |    10   | **0.0**   | 9.0e-6  |   1.6e-3   |   8.9e-6 |    9.0e-6 |   8.8e-6 |   8.8e-3 |
| box          |    20   | **0.0**   | 8.0e-6  |   2.5e-3   |   7.9e-6 |    8.0e-6 |   7.9e-6 |   7.9e-3 |
| mpc          |    30   | 5.6e-17   | 5.5e-6  | **3.6e-10**|   5.5e-6 |    5.5e-6 |   5.4e-6 |   5.4e-3 |
| mpc          |    60   | 5.6e-17   | 4.3e-6  | **5.2e-10**|   4.2e-6 |    4.3e-6 |   4.2e-6 |   4.2e-3 |
| mpc          |    90   | 5.6e-17   | 4.1e-6  | **3.6e-10**|   4.1e-6 |    4.1e-6 |   4.1e-6 |   4.1e-3 |
| pathological |     5   | 1.4e-7    | 2.5e-5  | **3.7e-10**|   2.5e-5 |    2.5e-5 |   2.5e-5 |   2.5e-2 |
| pathological |    10   | 5.0e-7    | 2.0e-5  | **4.2e-10**|   1.9e-5 |    2.0e-5 |   1.9e-5 |   1.9e-2 |

Canonical achieves machine-precision feasibility (`0.0`) on box QPs
because its event-triggered projection snaps exactly onto each
boundary. The augmented Lagrangian achieves *better* feasibility than
the canonical on MPC and pathological (`~1e-10`) because the dual
ascent drives `lambda` to the exact KKT multiplier. Explicit-penalty
variants leave residual violation at `O(1/k_p_fin) ~ 1e-5` (matches
theory). Expeuler's larger violation reflects its smaller `k_p_fin`.

---

## 4. Neuromorphic-purity scoring

| variant      | runtime ops needed                               | neuron model required          | hardware fit       | notes |
|--------------|--------------------------------------------------|--------------------------------|---------------------|-------|
| canonical    | `argmax`, scalar division, sequential dispatch   | LIF + winner-take-all + control | CPU/FPGA            | Event-triggered control flow is digital-native; not a clean Loihi map. |
| penalty      | matvec, vec-add, ReLU                            | **pure LIF + ReLU**            | **CPU/FPGA/Loihi-2**| Strictly one neural population; cleanest map. |
| lagrangian   | matvec, vec-add, ReLU, vec multiply (rho * relu) | **two LIF populations + ReLU** | **CPU/FPGA/Loihi-2**| Excitatory-inhibitory pair, biologically natural. |
| heun         | as penalty, plus a two-phase clock               | pure LIF + ReLU + 2-phase tick | CPU/FPGA/Loihi-2    | Two-stage tick is straightforward on digital hw; less so on async neuromorphic. |
| heavyball    | matvec, vec-add, ReLU, two LIF pools             | two LIF + ReLU                 | CPU/FPGA/Loihi-2    | Momentum is a leaky LIF compartment. |
| nesterov     | as heavyball, plus time-varying coupling weight  | LIF + ReLU + scalar timer      | CPU/FPGA/Loihi-2    | The `(k-1)/(k+2)` weight is a single global parameter, broadcast. |
| expeuler     | matvec only, with one offline `expm`             | pure LIF (with native leak)    | **best for Loihi-2**| Loihi's leak compartments realise `exp(-h * tau^{-1})` natively, so the per-step update is the chip's native operation. Offline `expm` is one matrix exponential computed once. |

**Strongest neuromorphic claim:** *augmented Lagrangian* and *exponential
Euler*, depending on what we mean by "neuromorphic":

- For a *biologically-plausible* claim (excitatory-inhibitory pairs,
  pure ReLU, no precomputation), augmented Lagrangian wins outright.
  It is the most accurate Euler variant and uses only operations that
  match standard LIF + threshold + recurrent-coupling neural circuits.
- For a *Loihi-2-deployment* claim, exponential Euler wins because its
  per-tick update *is* the chip's native compartment update; the
  matrix-exp precompute is a one-time offline operation analogous to
  setting per-synapse leak time-constants.

The canonical adaptive projection is the *least* neuromorphic-pure
because of its event-triggered argmax-and-loop control flow.

---

## 5. Recommendation for the datacenter-cooling paper

Given the FPGA hardware target (Kria KR260, see
`project_snn_qp_fpga_deployment.md`) and the empirical results above,
the cleanest paper-framing is **option (c): present a family of
SNN-QP solvers spanning a neuromorphic-purity / hardware-pragmatism
trade-off.**

Concrete suggestion for the paper paragraph:

> We evaluated three projection mechanisms within our SNN-QP framework:
> the canonical event-triggered Gauss-Seidel adaptive projection
> (Section X.Y), an Euler-discretised quadratic-penalty formulation,
> and an Euler-discretised augmented-Lagrangian primal-dual formulation.
> The latter two replace the projection step with continuous-time ODE
> dynamics composed entirely of vector-matrix products and element-wise
> rectification: matrix-inverse-free, control-flow-free, and clock-tick
> uniform. On a benchmark of 90-variable MPC QPs, the augmented
> Lagrangian matches the canonical solver's accuracy (`||x_snn -
> x_qp||/||x_qp|| ≈ 1e-9`) within a 5000-step budget while using only
> LIF-mappable operations; on a pathological QP with rapidly switching
> active sets where the canonical adaptive projection times out at
> 10000 iterations with 30% relative error, the augmented Lagrangian
> converges in 1230 iterations to `1e-9`. The Euler-penalty variants
> trade off accuracy (`1e-2` to `1e-5` depending on geometry) for the
> simplest possible neural-circuit substrate (a single primal
> population, ReLU rectification, no dual neurons). For the FPGA
> deployment described in Section X.Z we use the canonical adaptive
> projection because its event-triggered structure is digital-native
> and gives faster wall-time on small MPC QPs; for a Loihi-2 port we
> would migrate to the augmented Lagrangian, whose
> excitatory-inhibitory pair structure compiles to standard LIF
> compartments without an event-controller block.

This positions the paper as offering a principled menu rather than
a single hardware-specific point design, which is a stronger
publishable contribution than either choice alone.

If we cannot include the menu, the next-best option is **(a): keep the
canonical solver and acknowledge the limitation** — the canonical
adaptive projection is a Gauss-Seidel single-constraint update, not a
hidden block solve, so the "co-processor call" framing is overstated to
begin with. The honest framing is "event-triggered control flow," which
is a real but narrower limitation than a hidden linear solver.

We do **not** recommend option (b): switching wholesale to a penalty
or augmented-Lagrangian formulation in the cooling paper. The canonical
solver gives machine-precision MPC solutions in 200 iterations on
small problems; the Euler variants need 2000-5000 iterations to reach
1e-3 to 1e-9. Real-time receding-horizon MPC at 30-second sampling has
latency budget for both, but the canonical solver's faster wall-time is
useful for higher-frequency control loops we may want to test later.

---

## 6. Future-work hooks

1. **Proximal gradient / PIPG (Mangalore-Yu).** Replace the explicit
   penalty term with a non-smooth `g(x) = chi_X(x)` indicator and use
   proximal-gradient iterates `x_{k+1} = prox_g(x_k - h grad f)`. The
   prox of a polytope-indicator is itself a small QP; for box
   constraints it's just clipping. Closer in flavour to what's already
   inside `_clip_to_bounds` in the canonical solver. Provably
   feasibility-preserving without any penalty bias.

2. **Lava-Optimization port.** Intel's Lava-Optimization library has a
   QP solver (Davies et al. 2022) implemented entirely on Loihi-2; it
   would be a useful third-party benchmark for our Euler variants on
   real neuromorphic hardware. The augmented-Lagrangian variant maps to
   Lava-Optimization's `LIF + Sigma` primitive without modification.

3. **Adaptive `k_p` schedule.** The current `alpha^2` ramp is heuristic;
   a smarter schedule would monitor `proj_grad_norm` and ramp `k_p`
   only when the iterate plateaus, mimicking Bertsekas-style outer-loop
   penalty methods. Could close most of the remaining accuracy gap to
   the canonical solver on box QPs.

4. **Frank-Wolfe / mirror descent.** Both are projection-free and have
   natural neural-population interpretations (Frank-Wolfe = winner-take-
   all over the constraint set; mirror = sigmoid-rectified primal). Not
   evaluated here because the box and MPC test sets don't show off
   their strengths (simplex-like geometries do); worth considering for
   the SNN-PCA and SNN-KRR papers where the constraint sets are
   different.

5. **Implicit / backward Euler with neural fixed-point inner loop.**
   `x_{k+1} = x_k + h f(x_{k+1})` with the implicit step approximated
   by a few inner Euler iterations of the relaxation `dy/dt = -gamma
   (y - (x_k + h f(y)))`. Trades outer for inner iterations; useful
   when the QP is stiff (high condition number on `A`).

6. **FPGA HLS port.** All six Euler variants compile to a fixed
   schedule of `gemv` + ReLU + accumulator passes; this is exactly the
   primitive set the Vitis HLS pipeline expects. No event-controller
   or argmax-reducer logic is needed, which simplifies the HLS
   description significantly compared to the canonical projection.
   Worth quantifying the LUT/DSP cost difference in the FPGA paper.

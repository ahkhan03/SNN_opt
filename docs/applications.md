# Applications of `snn_opt`

`snn_opt` is the canonical solver behind the **SNN-X** publication series — a
sequence of papers that recast classical machine-learning problems as
constrained convex programs and show they can be solved by the same SNN
dynamics. It also underlies several applied projects (model-predictive
control, traffic-signal optimization, image denoising). This page is the
short-form catalogue; each entry links back to the full paper or repository.

## SNN-X series

| Paper | Problem class | Headline result |
|---|---|---|
| **SNN-SVM** | Box-constrained QP (SVM dual) | 99.3% accuracy, matches `scikit-learn` to 4 decimals on the standard SVM benchmarks. |
| **SNN-LinReg** | Equality-constrained least squares | Reproduces `scipy.linalg.lstsq` to machine precision on four constraint variants. |
| **SNN-CF** | ALS for collaborative filtering | +3.9% RMSE improvement over standard ALS on MovieLens-100K via the constrained formulation. |
| **SNN-PCA** | Sphere-constrained quadratic (eigenproblem) | Eigenvalue error ≤ 4.4e-12; "projected ascent" formulation lifts Mancoo equivalence to the non-convex setting. *Under review at IEEE TNNLS.* |
| **SNN-Procrustes** | Orthogonal Procrustes | Rotation error < 1e-12 on 3-D point-cloud registration. |
| **SNN-KRR** | Constrained kernel ridge regression | Hessian condition number reduced 100× via a reformulation that's natural in the SNN form. *Under review at IEEE TCAS-I.* |
| **SNN-Ridge** | Ridge regression with four constraint variants | Closed-form-precision agreement; serves as the cleanest tutorial application. |
| **SNN-TDSVM** | Two-timescale spiking solver for TDSVM | Non-trivial convex application with a custom two-timescale schedule. *Under review at* Neural Networks. |
| **SNN-Norm** | Homeostatic normalization for SNNs | Asymmetric dependency: NoNorm costs 3.5pp in ANNs but 76pp in SNNs; HN matches BatchNorm without batch statistics. *Under review.* |

The thread connecting all nine: each problem is reformulated as

$$
\min_x\ \tfrac12 x^\top A x + b^\top x \quad\text{s.t.}\quad C x + d \le 0
$$

and dropped into the same `snn_opt.solve_qp(A, b, C, d, ...)` call. The
solver itself is unchanged across applications — only the formulation
machinery and the problem-specific reductions vary.

## Applied projects

Beyond the methodology series, the same solver is the inner loop for several
applied research projects:

- **SNN-EnergyPlus** — Receding-horizon control of a building cooling loop.
  An SNN-equivalent QP at every control step drives an EnergyPlus simulation
  via the Python API. SNN solutions match CVXPY to 6 significant figures
  while staying within the millisecond budget.
- **Traffic-signal SNN** — SUMO-in-the-loop optimization of phase timings
  with capacity and pedestrian-wait constraints. The warm-started solver
  (Figure 3 of the README) scales naturally to the rolling horizon.
- **Neuromorphic image denoising** — `snn_opt` solves a per-patch
  total-variation QP across an image grid; benchmarked at 128×128 against
  classical TV denoising.

## How to add your own application

The recommended pattern for a new application is to write a small
*formulation module* that exposes a single `build_problem(...)` function
returning `(A, b, C, d, x0)`, plus any post-processing needed to map the
solver output back to the original problem variables. The downstream
solve is then:

```python
from snn_opt import solve_qp
from my_app import build_problem

A, b, C, d, x0 = build_problem(my_inputs)
result = solve_qp(A, b, C, d, x0)
```

Each of the SNN-X papers above is structured this way; their formulation
modules are the most useful starting templates.

## Citing the application series

A complete BibTeX file with all SNN-X paper entries lives in
[`docs/snn_x_series.bib`](snn_x_series.bib) *(populated as papers reach
final accepted form)*. For now, please cite the relevant paper directly
when applicable, and `snn_opt` itself via the [`CITATION.cff`](../CITATION.cff)
entry at the repository root.

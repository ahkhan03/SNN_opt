# Applications of `snn_opt`

`snn_opt` was developed in support of an ongoing research program that
formulates classical machine-learning and control problems as constrained
convex programs and solves them with the same SNN dynamics. This page is the
short-form catalogue; entries are added as the corresponding work appears in
print.

## Published

- **Khan, Mohammed & Li (2025)** — *Portfolio Optimization: A Neurodynamic
  Approach Based on Spiking Neural Networks.* **Biomimetics**, 10(12):808.
  [doi:10.3390/biomimetics10120808](https://doi.org/10.3390/biomimetics10120808).
  The portfolio-selection problem is recast as a constrained QP and solved
  by the spiking dynamics implemented in this repository, demonstrating
  that SNN solutions match conventional convex solvers within numerical
  tolerance while exposing the per-asset constraint activations as
  spike events.

A broader survey of SNN training and hardware accompanies this work:
**Khan et al. (2025)** — *Spiking Neural Networks: A Comprehensive Survey
of Training Methodologies, Hardware Implementations and Applications,*
**Artificial Intelligence Science and Engineering**.
[doi:10.23919/AISE.2025.000013](https://doi.org/10.23919/AISE.2025.000013).

## In preparation

Additional applications of the framework — covering further classical
ML reductions and control problems — are at various stages of preparation
and review. Entries will be added here as they are accepted for
publication.

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

## Citing the framework

Please cite `snn_opt` itself via the [`CITATION.cff`](../CITATION.cff)
entry at the repository root, and the relevant application paper above
when applicable. As further papers in the program reach publication they
will be appended here with full BibTeX entries.

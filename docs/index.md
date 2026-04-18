# `snn_opt` documentation

This directory hosts the documentation for `snn_opt`, the spiking neural
network solver for constrained convex optimization. The repository
[`README`](../README.md) gives the short tour; the pages here are the
longer-form companion.

## Pages

- [**Theory**](theory.md) — Full derivation of the SNN/convex-optimization
  equivalence, the eigenvalue-based step-size analysis, projection
  geometry, and convergence criteria. Begins from LIF dynamics and arrives
  at the projected-gradient algorithm implemented in
  [`src/snn_opt/solver.py`](../src/snn_opt/solver.py).
- [**Applications**](applications.md) — One-page catalogue of the SNN-X
  publication series (PCA, Ridge, TDSVM, Norm, SVM, CF, KRR, Procrustes,
  LinReg) and the applied projects that use this solver as their inner
  loop (EnergyPlus MPC, SUMO traffic, image denoising).
- [**API reference**](api.md) — Hand-curated reference for every public
  symbol exported from `snn_opt`. The repository is small enough that this
  is faster than auto-generated docs.
- [**Benchmarks**](../benchmarks/README.md) — How the convergence,
  spike-raster, and warm-start figures embedded in the README are
  produced; how to regenerate them; and what each one shows.

## Companion website

A complementary, more accessibly-written companion site lives at
[**snn.ahkhan.me**](https://snn.ahkhan.me). The repository here is the
canonical record (code, math, figures, paper-by-paper applications); the
site is the place to send students and curious researchers who want a
gentler entry point.

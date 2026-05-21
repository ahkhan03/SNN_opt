#!/usr/bin/env python3
"""
Apples-to-apples solver benchmark: SNN-QP vs OSQP / CVXPY / SciPy.

Compares wall-clock and achieved accuracy for the SNN-QP solver (compiled C
backend and pure-Python lean path) against established QP solvers, on the
problem family the SNN-X papers target:

    minimize    (1/2) x^T A x + b^T x
    subject to  C x + d <= 0   [, lower <= x <= upper]

Test problems use a Hessian with an *explicitly controlled condition number*
(eigenvalues log-spaced over [1, kappa], random orthogonal eigenbasis). This
matters: SNN-QP is a first-order projected-gradient method, so its iteration
count scales with kappa. Benchmarking on generic random Hessians (kappa ~ 40n,
uncontrolled) would conflate the solver with an ill-conditioned synthetic test
distribution. Two studies isolate the two axes:

    Study 1  n-sweep   -- fixed kappa, growing n   (fair same-accuracy timing)
    Study 2  kappa-sweep -- fixed n, growing kappa (first-order scaling law)

Fairness protocol
-----------------
* Reference optimum f* = smallest feasible objective among high-accuracy solves
  (CVXPY+Clarabel interior-point and OSQP polished at eps=1e-9), cross-checked.
* Every solver runs with ONE fixed tolerance setting across all problems (no
  per-problem tuning); each solver's achieved accuracy (objective gap to f*,
  max constraint violation) is reported next to its time.
* Setup and solve timed separately. SNN-QP setup = SNNSolver.__init__ (the
  k0 = lambda_max(A) eigensolve + constraint-norm precompute); OSQP setup =
  .setup() (KKT factorisation); CVXPY setup = canonicalisation.
* Wall-time = median of repeated cold solves after one warm-up.

Outputs (analysis/results/):
  bench_vs_libraries.json        -- raw per-instance records
  bench_vs_libraries_summary.md  -- aggregated tables, grouped by study

Run:  python analysis/bench_vs_libraries.py
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize

import cvxpy as cp
import osqp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from snn_opt import (  # noqa: E402
    ConvergenceConfig, OptimizationProblem, SNNSolver, SolverConfig,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"

GAP_TARGET = 1e-6
VIOL_TARGET = 1e-6
SOLVER_ORDER = ["snn-qp-c", "snn-qp-py", "osqp", "cvxpy-osqp", "scipy-slsqp"]


# ===========================================================================
# Problem battery -- Hessian with explicitly controlled condition number
# ===========================================================================
def conditioned_qp(n, kappa, m, seed, *, box=False, study="", key=0):
    """QP whose Hessian has eigenvalues log-spaced over [1, kappa] in a random
    orthogonal eigenbasis -- so kappa(A) == kappa exactly."""
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.standard_normal((n, n)))
    eig = np.logspace(0.0, np.log10(kappa), n)
    A = (U * eig) @ U.T
    A = 0.5 * (A + A.T)
    b = rng.standard_normal(n)
    if m > 0:
        C = rng.standard_normal((m, n))
        d = -np.ones(m) - np.maximum(C @ np.zeros(n), 0.0)   # x = 0 feasible
    else:
        C = np.zeros((0, n))
        d = np.zeros(0)
    lo, up = (-1.0, 1.0) if box else (None, None)
    tag = "box" if box else "qp"
    return dict(name=f"{tag} n={n} kappa={kappa} s{seed}",
                A=A, b=b, C=C, d=d, x0=np.zeros(n),
                lower=lo, upper=up, n=n, m=m, kappa=kappa, box=box,
                study=study, key=key)


def battery():
    out = []
    # Study 1 -- n-sweep at a fixed, benign condition number.
    for n in (10, 25, 50, 100, 200, 400):
        for s in (1, 2, 3):
            out.append(conditioned_qp(n, 30, n // 2, s,
                                      study="n-sweep", key=n))
    # Study 2 -- kappa-sweep at a fixed problem size.
    for kappa in (3, 10, 30, 100, 300, 1000):
        for s in (1, 2, 3):
            out.append(conditioned_qp(100, kappa, 50, s,
                                      study="kappa-sweep", key=kappa))
    # Study 3 -- box-constrained (SVM-dual-like), benign conditioning.
    for n in (25, 75):
        for s in (1, 2):
            out.append(conditioned_qp(n, 30, max(1, n // 4), s, box=True,
                                      study="box", key=n))
    return out


# ===========================================================================
# Accuracy helpers
# ===========================================================================
def objective(p, x):
    return float(0.5 * x @ p["A"] @ x + p["b"] @ x)


def max_violation(p, x):
    v = 0.0
    if p["m"] > 0:
        v = max(v, float(np.max(np.maximum(p["C"] @ x + p["d"], 0.0))))
    if p["lower"] is not None:
        v = max(v, float(np.max(np.maximum(p["lower"] - x, 0.0))))
    if p["upper"] is not None:
        v = max(v, float(np.max(np.maximum(x - p["upper"], 0.0))))
    return v


def accuracy(p, x, f_star):
    gap = abs(objective(p, x) - f_star) / max(abs(f_star), 1.0)
    return gap, max_violation(p, x)


# ===========================================================================
# Reference optimum (solver-independent)
# ===========================================================================
def reference(p):
    candidates = []
    prob, x = _build_cvxpy(p)
    prob.solve(solver=cp.CLARABEL)
    if x.value is not None:
        xc = np.asarray(x.value, float)
        candidates.append(("clarabel", objective(p, xc), max_violation(p, xc)))

    P, q, Am, lo, up = _build_osqp(p)
    m = osqp.OSQP()
    m.setup(P=P, q=q, A=Am, l=lo, u=up, verbose=False,
            eps_abs=1e-9, eps_rel=1e-9, max_iter=200_000, polish=True)
    res = m.solve()
    xo = np.asarray(res.x, float)
    if np.all(np.isfinite(xo)):
        candidates.append(("osqp", objective(p, xo), max_violation(p, xo)))

    feas = [(lbl, f) for (lbl, f, v) in candidates if v < 1e-7]
    if not feas:
        raise RuntimeError(f"no feasible reference for {p['name']}")
    f_star = min(f for _, f in feas)
    spread = max(f for _, f in feas) - min(f for _, f in feas)
    return f_star, spread


# ===========================================================================
# Per-solver builders
# ===========================================================================
def _build_cvxpy(p):
    x = cp.Variable(p["n"])
    obj = cp.Minimize(0.5 * cp.quad_form(x, cp.psd_wrap(p["A"])) + p["b"] @ x)
    cons = []
    if p["m"] > 0:
        cons.append(p["C"] @ x + p["d"] <= 0)
    if p["lower"] is not None:
        cons.append(x >= p["lower"])
    if p["upper"] is not None:
        cons.append(x <= p["upper"])
    return cp.Problem(obj, cons), x


def _build_osqp(p):
    """OSQP form: min 1/2 x'Px + q'x  s.t.  l <= A x <= u."""
    n = p["n"]
    P = sp.csc_matrix(np.triu(p["A"]))
    blocks, lo, up = [], [], []
    if p["m"] > 0:
        blocks.append(sp.csc_matrix(p["C"]))
        lo.append(np.full(p["m"], -np.inf))
        up.append(-p["d"])
    if p["box"]:
        blocks.append(sp.identity(n, format="csc"))
        lo.append(np.full(n, p["lower"]))
        up.append(np.full(n, p["upper"]))
    Am = sp.vstack(blocks, format="csc")
    return P, p["b"], Am, np.concatenate(lo), np.concatenate(up)


# ===========================================================================
# Solver runners
# ===========================================================================
def _median(ts):
    return float(np.median(ts))


def run_snn(p, backend, reps):
    conv = ConvergenceConfig(proj_grad_tol=1e-6, obj_rel_tol=1e-9,
                             feasibility_tol=1e-7)
    cfg = SolverConfig(max_iterations=100_000, backend=backend,
                       record_trajectory=False, constraint_tol=1e-8,
                       lower_bound=p["lower"], upper_bound=p["upper"],
                       convergence=conv)
    prob = OptimizationProblem(A=p["A"], b=p["b"], C=p["C"], d=p["d"])
    setup_ts, solve_ts, res = [], [], None
    for rep in range(reps + 1):
        t0 = time.perf_counter()
        solver = SNNSolver(prob, cfg)
        t1 = time.perf_counter()
        res = solver.solve(p["x0"])
        t2 = time.perf_counter()
        if rep > 0:
            setup_ts.append(t1 - t0)
            solve_ts.append(t2 - t1)
    return dict(setup_t=_median(setup_ts), solve_t=_median(solve_ts),
                x=np.asarray(res.final_x, float), iters=res.iterations_used,
                status="converged" if res.converged else "max_iter")


def run_osqp(p, reps):
    P, q, Am, lo, up = _build_osqp(p)
    setup_ts, solve_ts, res = [], [], None
    for rep in range(reps + 1):
        m = osqp.OSQP()
        t0 = time.perf_counter()
        m.setup(P=P, q=q, A=Am, l=lo, u=up, verbose=False,
                eps_abs=1e-8, eps_rel=1e-8, max_iter=50_000, polish=True)
        t1 = time.perf_counter()
        res = m.solve()
        t2 = time.perf_counter()
        if rep > 0:
            setup_ts.append(t1 - t0)
            solve_ts.append(t2 - t1)
    return dict(setup_t=_median(setup_ts), solve_t=_median(solve_ts),
                x=np.asarray(res.x, float), iters=int(res.info.iter),
                status=str(res.info.status))


def run_cvxpy(p, reps):
    prob, x = _build_cvxpy(p)
    kw = dict(solver=cp.OSQP, eps_abs=1e-8, eps_rel=1e-8, polish=True,
              warm_start=False, verbose=False)
    t0 = time.perf_counter()
    prob.solve(**kw)
    first = time.perf_counter() - t0
    solve_ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        prob.solve(**kw)
        solve_ts.append(time.perf_counter() - t0)
    solve_t = _median(solve_ts)
    return dict(setup_t=max(first - solve_t, 0.0), solve_t=solve_t,
                x=np.asarray(x.value, float),
                iters=getattr(prob.solver_stats, "num_iters", None),
                status=str(prob.status))


def run_slsqp(p, reps):
    A, b, C, d = p["A"], p["b"], p["C"], p["d"]
    fun = lambda x: 0.5 * x @ A @ x + b @ x          # noqa: E731
    jac = lambda x: A @ x + b                        # noqa: E731
    cons = []
    if p["m"] > 0:
        cons.append(dict(type="ineq", fun=lambda x: -(C @ x + d),
                         jac=lambda x: -C))
    bounds = [(p["lower"], p["upper"])] * p["n"] if p["box"] else None
    opts = dict(ftol=1e-12, maxiter=300)
    solve_ts, res = [], None
    for _ in range(reps):
        t0 = time.perf_counter()
        res = minimize(fun, p["x0"], jac=jac, method="SLSQP",
                       constraints=cons, bounds=bounds, options=opts)
        solve_ts.append(time.perf_counter() - t0)
    return dict(setup_t=0.0, solve_t=_median(solve_ts),
                x=np.asarray(res.x, float), iters=int(res.nit),
                status="success" if res.success else f"fail:{res.message}")


# ===========================================================================
# Main
# ===========================================================================
def reps_for(p, solver):
    big = p["n"] >= 200 or p["kappa"] >= 300
    if solver == "snn-qp-py":
        return 1 if big else 2
    if solver == "scipy-slsqp":
        return 1 if big else 2
    return 3 if big else 6


def main():
    problems = battery()
    records = []
    print(f"Benchmarking {len(problems)} problems "
          f"(target: gap<={GAP_TARGET:.0e}, viol<={VIOL_TARGET:.0e})\n")

    for p in problems:
        f_star, spread = reference(p)
        rec = dict(name=p["name"], n=p["n"], m=p["m"], kappa=p["kappa"],
                   box=p["box"], study=p["study"], key=p["key"],
                   f_star=f_star, ref_spread=spread, solvers={})
        print(f"{p['name']:<26} f*={f_star:+.5e}  ref-spread={spread:.1e}")

        runners = {
            "snn-qp-c":   lambda p=p: run_snn(p, "c", reps_for(p, "snn-qp-c")),
            "snn-qp-py":  lambda p=p: run_snn(p, "python",
                                              reps_for(p, "snn-qp-py")),
            "osqp":       lambda p=p: run_osqp(p, reps_for(p, "osqp")),
            "cvxpy-osqp": lambda p=p: run_cvxpy(p, reps_for(p, "cvxpy-osqp")),
            "scipy-slsqp": lambda p=p: run_slsqp(p,
                                                 reps_for(p, "scipy-slsqp")),
        }
        for sname, runner in runners.items():
            try:
                r = runner()
                gap, viol = accuracy(p, r["x"], f_star)
                rec["solvers"][sname] = dict(
                    setup_ms=r["setup_t"] * 1e3, solve_ms=r["solve_t"] * 1e3,
                    total_ms=(r["setup_t"] + r["solve_t"]) * 1e3,
                    gap=gap, viol=viol, iters=r["iters"], status=r["status"],
                    hit_target=bool(gap <= GAP_TARGET and viol <= VIOL_TARGET))
                tgt = "ok " if rec["solvers"][sname]["hit_target"] else "MISS"
                print(f"    {sname:<13} solve={r['solve_t']*1e3:9.3f} ms  "
                      f"gap={gap:.1e} viol={viol:.1e} [{tgt}] "
                      f"iters={r['iters']}")
            except Exception as exc:                  # noqa: BLE001
                rec["solvers"][sname] = dict(error=repr(exc))
                print(f"    {sname:<13} ERROR: {exc!r}")
        records.append(rec)
        print()

    RESULTS_DIR.mkdir(exist_ok=True)
    raw_path = RESULTS_DIR / "bench_vs_libraries.json"
    raw_path.write_text(json.dumps(records, indent=2))
    _write_summary(records, RESULTS_DIR / "bench_vs_libraries_summary.md")
    print(f"raw    -> {raw_path}")
    print(f"summary-> {RESULTS_DIR / 'bench_vs_libraries_summary.md'}")
    _print_headline(records)
    return 0


# ---------------------------------------------------------------------------
# Aggregation + summary
# ---------------------------------------------------------------------------
def _aggregate(records):
    """Group by (study, key); return median of each metric over seeds."""
    groups = defaultdict(list)
    for r in records:
        groups[(r["study"], r["key"])].append(r)
    agg = {}
    for gkey, recs in groups.items():
        per_solver = {}
        for s in SOLVER_ORDER:
            cells = [r["solvers"].get(s) for r in recs]
            cells = [c for c in cells if c and "error" not in c]
            if not cells:
                per_solver[s] = None
                continue
            per_solver[s] = dict(
                solve_ms=float(np.median([c["solve_ms"] for c in cells])),
                total_ms=float(np.median([c["total_ms"] for c in cells])),
                gap=float(np.median([c["gap"] for c in cells])),
                iters=float(np.median([c["iters"] for c in cells
                                       if c["iters"] is not None] or [0])),
                hit=all(c["hit_target"] for c in cells))
        agg[gkey] = dict(n=recs[0]["n"], kappa=recs[0]["kappa"],
                         m=recs[0]["m"], solvers=per_solver,
                         n_seeds=len(recs))
    return agg


def _write_summary(records, path):
    agg = _aggregate(records)
    L = ["# SNN-QP vs OSQP / CVXPY / SciPy -- benchmark summary", "",
         f"Common-accuracy target: objective gap <= {GAP_TARGET:.0e}, "
         f"constraint violation <= {VIOL_TARGET:.0e}.",
         "Times are the median over seeds of repeated cold solves (ms). "
         "Hessian condition number is controlled exactly.", ""]

    def time_table(study, axis_label, axis_of):
        keys = sorted(k for (st, k) in agg if st == study)
        rows = [f"## {study}: solve time (ms), median over seeds", "",
                f"| {axis_label} | " + " | ".join(SOLVER_ORDER)
                + " | snn-c / osqp |",
                "|" + "---|" * (len(SOLVER_ORDER) + 2)]
        for k in keys:
            g = agg[(study, k)]
            cells = []
            for s in SOLVER_ORDER:
                sd = g["solvers"][s]
                cells.append(f"{sd['solve_ms']:.2f}" if sd else "--")
            sc, oq = g["solvers"]["snn-qp-c"], g["solvers"]["osqp"]
            ratio = (f"{sc['solve_ms'] / oq['solve_ms']:.0f}x"
                     if sc and oq and oq["solve_ms"] > 0 else "--")
            rows.append(f"| {axis_of(g)} | " + " | ".join(cells)
                        + f" | {ratio} |")
        rows.append("")
        return rows

    L += time_table("n-sweep", "n (kappa=30)", lambda g: g["n"])
    L += time_table("kappa-sweep", "kappa (n=100)", lambda g: g["kappa"])
    L += time_table("box", "n (box, kappa=30)", lambda g: g["n"])

    # kappa-sweep iteration scaling -- the first-order signature
    L += ["## kappa-sweep: iteration count (median over seeds)", "",
          "| kappa | snn-qp-c iters | osqp iters | snn solve (ms) "
          "| osqp solve (ms) |", "|---|---|---|---|---|"]
    for k in sorted(kk for (st, kk) in agg if st == "kappa-sweep"):
        g = agg[("kappa-sweep", k)]
        sc, oq = g["solvers"]["snn-qp-c"], g["solvers"]["osqp"]
        L.append(f"| {k} | {sc['iters']:.0f} | {oq['iters']:.0f} "
                 f"| {sc['solve_ms']:.2f} | {oq['solve_ms']:.2f} |")
    L.append("")

    # accuracy confirmation
    L += ["## Achieved accuracy -- did every solver hit the target?", "",
          "| group | " + " | ".join(SOLVER_ORDER) + " |",
          "|" + "---|" * (len(SOLVER_ORDER) + 1)]
    for (study, k) in sorted(agg):
        g = agg[(study, k)]
        cells = []
        for s in SOLVER_ORDER:
            sd = g["solvers"][s]
            cells.append(("ok" if sd["hit"] else f"MISS({sd['gap']:.0e})")
                         if sd else "--")
        L.append(f"| {study} {k} | " + " | ".join(cells) + " |")
    L += ["",
          "Setup vs solve: 'solve' is the warm/amortised-setup cost; the raw "
          "JSON also has per-solver setup_ms and total_ms (one-shot cost).",
          "SNN-QP setup = the k0 = lambda_max(A) eigensolve."]
    path.write_text("\n".join(L))


def _print_headline(records):
    print("\n=== headline (geomean over instances where both hit target) ===")
    base = "snn-qp-c"
    for other in ["snn-qp-py", "osqp", "cvxpy-osqp", "scipy-slsqp"]:
        ratios = []
        for r in records:
            a, b = r["solvers"].get(base), r["solvers"].get(other)
            if (a and b and "error" not in a and "error" not in b
                    and a["hit_target"] and b["hit_target"]
                    and a["solve_ms"] > 0):
                ratios.append(b["solve_ms"] / a["solve_ms"])
        if ratios:
            gm = float(np.exp(np.mean(np.log(ratios))))
            rel = (f"{gm:.2f}x ({base} faster)" if gm >= 1
                   else f"{1 / gm:.2f}x ({other} faster)")
            print(f"  solve  {other:<12} vs {base}: {rel}  (n={len(ratios)})")


if __name__ == "__main__":
    sys.exit(main())

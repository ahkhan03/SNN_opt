#!/usr/bin/env python3
"""
Benchmark: closed-form projection vs Euler-integrated projection variants.

Test sets
---------
1. Small synthetic MPC QPs    (5/10/15 zones x horizon-10, 5 seeds each)
2. Box-constrained random QPs (n in {5, 10, 20}, 5 seeds each)
3. Pathological "rapid active-set switching" QP

For each (test, problem-size, seed) we solve with all 7 variants and record:
- iter_at_termination
- rel_err to CVXPY (||x - x_cvx||_2 / ||x_cvx||_2)
- max constraint violation at termination
- wall time
- iter to reach rel_err < 1e-2 (if attained within budget)

Output is dumped to analysis/results/benchmark_raw.npz and
analysis/results/benchmark_summary.yaml.
"""

import sys
import time
import yaml
import numpy as np
from pathlib import Path
from datetime import datetime

import cvxpy as cp
from scipy.linalg import solve_discrete_are

from snn_opt import (
    solve_qp,
    solve_qp_penalty,
    solve_qp_lagrangian,
    solve_qp_heun_penalty,
    solve_qp_heavyball_penalty,
    solve_qp_nesterov_penalty,
    solve_qp_expeuler_penalty,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# --- variant registry -------------------------------------------------------

VARIANTS = [
    ("canonical",  lambda H, f, C, d, x0, mi: solve_qp(
        H, f, C, d, x0, max_iterations=mi, verbose=False)),
    ("penalty",    lambda H, f, C, d, x0, mi: solve_qp_penalty(
        H, f, C, d, x0, max_iterations=mi, verbose=False)),
    ("lagrangian", lambda H, f, C, d, x0, mi: solve_qp_lagrangian(
        H, f, C, d, x0, max_iterations=mi, verbose=False)),
    ("heun",       lambda H, f, C, d, x0, mi: solve_qp_heun_penalty(
        H, f, C, d, x0, max_iterations=mi, verbose=False)),
    ("heavyball",  lambda H, f, C, d, x0, mi: solve_qp_heavyball_penalty(
        H, f, C, d, x0, max_iterations=mi, verbose=False)),
    ("nesterov",   lambda H, f, C, d, x0, mi: solve_qp_nesterov_penalty(
        H, f, C, d, x0, max_iterations=mi, verbose=False)),
    ("expeuler",   lambda H, f, C, d, x0, mi: solve_qp_expeuler_penalty(
        H, f, C, d, x0, max_iterations=mi, verbose=False)),
]


# --- MPC QP generator (replicated from paper workspace) ---------------------

def build_prediction_matrices(A, B, N):
    n = A.shape[0]
    m = B.shape[1]
    S_x = np.zeros((N * n, n))
    S_u = np.zeros((N * n, N * m))
    A_pow = np.eye(n)
    for i in range(N):
        A_pow = A @ A_pow if i > 0 else A
        S_x[i*n:(i+1)*n, :] = A_pow
    AB_blocks = [B.copy()]
    for i in range(1, N):
        AB_blocks.append(A @ AB_blocks[-1])
    for i in range(N):
        for j in range(i + 1):
            S_u[i*n:(i+1)*n, j*m:(j+1)*m] = AB_blocks[i - j]
    return S_x, S_u


def generate_thermal_system(n_zones, n_controls, seed=42):
    rng = np.random.default_rng(seed)
    A = np.eye(n_zones) * rng.uniform(0.90, 0.97, n_zones)
    for i in range(n_zones - 1):
        c = rng.uniform(0.005, 0.02)
        A[i, i+1] = c; A[i+1, i] = c
    eigs = np.abs(np.linalg.eigvals(A))
    if eigs.max() >= 1.0:
        A *= 0.98 / eigs.max()
    B = np.zeros((n_zones, n_controls))
    for j in range(n_controls):
        center = int(j * n_zones / n_controls)
        for off in range(-1, 2):
            idx = center + off
            if 0 <= idx < n_zones:
                B[idx, j] = -rng.uniform(0.05, 0.15)
    return A, B


def generate_mpc_qp(A, B, N, rng):
    n = A.shape[0]; m = B.shape[1]
    Q = np.eye(n) * rng.uniform(0.1, 0.5)
    R = np.eye(m) * rng.uniform(0.05, 0.2)
    try:
        P = solve_discrete_are(A, B, Q, R)
    except Exception:
        P = Q.copy()
    x0 = rng.uniform(20, 26, n)
    x_ref = np.full(n, 22.0)
    u_min = np.full(m, 0.2)
    u_max = np.full(m, 1.0)

    S_x, S_u = build_prediction_matrices(A, B, N)
    Q_bar = np.kron(np.eye(N - 1), Q) if N > 1 else np.zeros((0, 0))
    if N > 1:
        Q_full = np.block([[Q_bar, np.zeros((Q_bar.shape[0], P.shape[1]))],
                           [np.zeros((P.shape[0], Q_bar.shape[1])), P]])
    else:
        Q_full = P
    R_bar = np.kron(np.eye(N), R)
    H = S_u.T @ Q_full @ S_u + R_bar
    H = 0.5 * (H + H.T) + 1e-8 * np.eye(H.shape[0])
    f = S_u.T @ Q_full @ (S_x @ x0 - np.tile(x_ref, N))

    # Box constraints on U
    Nu = N * m
    C_box = np.vstack([np.eye(Nu), -np.eye(Nu)])
    d_box = np.concatenate([-np.tile(u_max, N), np.tile(u_min, N)])
    return H, f, C_box, d_box


# --- random box-QP generator ------------------------------------------------

def generate_random_box_qp(n, seed):
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, n))
    A = M.T @ M / n + 0.05 * np.eye(n)
    b = rng.standard_normal(n)
    bound = rng.uniform(0.5, 1.5)
    C = np.vstack([np.eye(n), -np.eye(n)])
    d = np.concatenate([-np.ones(n) * bound, -np.ones(n) * bound])
    return A, b, C, d


# --- pathological QP: many redundant constraints, rapidly switching set -----

def generate_pathological_qp(n, seed):
    """
    Construct a QP whose unconstrained optimum is at a vertex of a polytope
    with many constraints active simultaneously, plus extra near-redundant
    constraints that are *almost* active. Designed to stress active-set logic.
    """
    rng = np.random.default_rng(seed)
    A = np.eye(n) + 0.01 * rng.standard_normal((n, n))
    A = A.T @ A + 0.01 * np.eye(n)
    b = -10.0 * np.ones(n)  # pushes optimum to the +ve corner

    # Box -1 <= x <= 1 (so unconstrained opt is far outside, all upper-box
    # constraints will be active at optimum)
    C_box = np.vstack([np.eye(n), -np.eye(n)])
    d_box = np.concatenate([-np.ones(n), -np.ones(n)])

    # Add 2*n random redundant half-space constraints that are *almost* active
    # near the corner but in random directions
    n_extra = 2 * n
    C_extra = rng.standard_normal((n_extra, n))
    C_extra /= np.linalg.norm(C_extra, axis=1, keepdims=True)
    d_extra = -np.abs(C_extra @ np.ones(n)) - rng.uniform(0.01, 0.1, n_extra)
    # so c_i^T (1,...,1) + d_i = 0 - eps -> just inside

    C = np.vstack([C_box, C_extra])
    d = np.concatenate([d_box, d_extra])
    return A, b, C, d


# --- solve helpers ----------------------------------------------------------

def cvxpy_solve(A, b, C, d):
    n = A.shape[0]
    x = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(x, A) + b @ x),
                      [C @ x + d <= 0])
    t0 = time.perf_counter()
    prob.solve(solver=cp.CLARABEL)
    dt = time.perf_counter() - t0
    return np.asarray(x.value), float(prob.value), dt


def measure_variant(name, fn, A, b, C, d, x0, x_star, max_iter, target_eps):
    t0 = time.perf_counter()
    try:
        r = fn(A, b, C, d, x0, max_iter)
    except Exception as e:
        return {
            "name": name, "iters": max_iter, "rel_err": float("nan"),
            "max_viol": float("nan"), "wall_ms": float("nan"),
            "iter_to_eps": None, "converged": False, "error": str(e),
        }
    dt = time.perf_counter() - t0
    x_norm = max(np.linalg.norm(x_star), 1e-12)
    rel_err = float(np.linalg.norm(r.final_x - x_star) / x_norm)
    max_viol = float(np.max(np.maximum(C @ r.final_x + d, 0.0))) \
        if C.shape[0] > 0 else 0.0

    # iter to reach eps (scan trajectory)
    iter_to_eps = None
    if r.X is not None and r.X.shape[0] > 0:
        rel_traj = np.linalg.norm(r.X - x_star[None, :], axis=1) / x_norm
        below = np.where(rel_traj <= target_eps)[0]
        if below.size > 0:
            iter_to_eps = int(below[0])
    return {
        "name": name, "iters": int(r.iterations_used),
        "rel_err": rel_err, "max_viol": max_viol, "wall_ms": dt * 1000.0,
        "iter_to_eps": iter_to_eps, "converged": bool(r.converged),
    }


# --- experiments ------------------------------------------------------------

def run_test_mpc(max_iter=5000, target_eps=1e-2):
    print("\n" + "=" * 70)
    print("Test 1: MPC QPs (datacenter-cooling generator)")
    print("=" * 70)
    sizes = [(5, 10), (10, 10), (15, 10)]   # (n_zones, horizon)
    n_seeds = 5
    rows = []
    for (n_zones, N) in sizes:
        n_controls = max(2, int(np.ceil(n_zones * 0.6)))
        Asys, Bsys = generate_thermal_system(n_zones, n_controls, seed=42)
        for seed in range(n_seeds):
            rng = np.random.default_rng(1000 + seed)
            H, f, C, d = generate_mpc_qp(Asys, Bsys, N, rng)
            x0 = np.zeros(H.shape[0])
            try:
                x_star, obj_star, t_cvx = cvxpy_solve(H, f, C, d)
            except Exception as e:
                print(f"  CVXPY failed on n_zones={n_zones} seed={seed}: {e}")
                continue
            for name, fn in VARIANTS:
                m = measure_variant(name, fn, H, f, C, d, x0, x_star,
                                    max_iter, target_eps)
                m.update({"test": "mpc", "n_zones": n_zones, "horizon": N,
                          "seed": seed, "n_vars": int(H.shape[0]),
                          "n_constraints": int(C.shape[0]), "cvxpy_ms": t_cvx*1000})
                rows.append(m)
            best_e = min(r["rel_err"] for r in rows[-len(VARIANTS):]
                         if not np.isnan(r["rel_err"]))
            print(f"  n_zones={n_zones} N={N} seed={seed}  best_rel_err={best_e:.2e}")
    return rows


def run_test_box(max_iter=5000, target_eps=1e-2):
    print("\n" + "=" * 70)
    print("Test 2: Random box-constrained QPs")
    print("=" * 70)
    sizes = [5, 10, 20]
    n_seeds = 5
    rows = []
    for n in sizes:
        for seed in range(n_seeds):
            A, b, C, d = generate_random_box_qp(n, seed)
            x0 = np.zeros(n)
            x_star, obj_star, t_cvx = cvxpy_solve(A, b, C, d)
            for name, fn in VARIANTS:
                m = measure_variant(name, fn, A, b, C, d, x0, x_star,
                                    max_iter, target_eps)
                m.update({"test": "box", "n": n, "seed": seed,
                          "n_vars": int(n), "n_constraints": int(C.shape[0]),
                          "cvxpy_ms": t_cvx*1000})
                rows.append(m)
            best_e = min(r["rel_err"] for r in rows[-len(VARIANTS):]
                         if not np.isnan(r["rel_err"]))
            print(f"  n={n} seed={seed}  best_rel_err={best_e:.2e}")
    return rows


def run_test_pathological(max_iter=10000, target_eps=1e-2):
    print("\n" + "=" * 70)
    print("Test 3: Pathological active-set switching QP")
    print("=" * 70)
    sizes = [5, 10]
    n_seeds = 3
    rows = []
    for n in sizes:
        for seed in range(n_seeds):
            A, b, C, d = generate_pathological_qp(n, seed)
            x0 = np.zeros(n)
            x_star, obj_star, t_cvx = cvxpy_solve(A, b, C, d)
            for name, fn in VARIANTS:
                m = measure_variant(name, fn, A, b, C, d, x0, x_star,
                                    max_iter, target_eps)
                m.update({"test": "pathological", "n": n, "seed": seed,
                          "n_vars": int(n), "n_constraints": int(C.shape[0]),
                          "cvxpy_ms": t_cvx*1000})
                rows.append(m)
            best_e = min(r["rel_err"] for r in rows[-len(VARIANTS):]
                         if not np.isnan(r["rel_err"]))
            print(f"  n={n} seed={seed}  best_rel_err={best_e:.2e}")
    return rows


# --- aggregator -------------------------------------------------------------

def aggregate(rows):
    """Group by (test, n_vars, variant) and report mean/std across seeds."""
    from collections import defaultdict
    groups = defaultdict(list)
    for r in rows:
        key = (r["test"], r.get("n_vars"), r.get("n_constraints"), r["name"])
        groups[key].append(r)
    summary = []
    for (test, n_vars, n_cons, name), grp in sorted(groups.items()):
        rel_errs = np.array([r["rel_err"] for r in grp], dtype=float)
        max_viols = np.array([r["max_viol"] for r in grp], dtype=float)
        iters = np.array([r["iters"] for r in grp], dtype=float)
        wall = np.array([r["wall_ms"] for r in grp], dtype=float)
        iters_to_eps = [r["iter_to_eps"] for r in grp]
        n_reached = sum(1 for v in iters_to_eps if v is not None)
        summary.append({
            "test": test, "n_vars": int(n_vars) if n_vars else None,
            "n_constraints": int(n_cons) if n_cons else None,
            "variant": name, "n_seeds": len(grp),
            "rel_err_mean": _f(np.nanmean(rel_errs)),
            "rel_err_std":  _f(np.nanstd(rel_errs)),
            "rel_err_median": _f(np.nanmedian(rel_errs)),
            "max_viol_mean": _f(np.nanmean(max_viols)),
            "iters_mean": _f(np.nanmean(iters)),
            "wall_ms_mean": _f(np.nanmean(wall)),
            "n_reached_eps": int(n_reached),
            "iter_to_eps_median": _f_median_int(iters_to_eps),
        })
    return summary


def _f(v):
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return None
    return float(v)


def _f_median_int(seq):
    nums = [v for v in seq if v is not None]
    if not nums:
        return None
    return int(np.median(nums))


# --- main ------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Projection-variant benchmark")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)
    rows_mpc = run_test_mpc()
    rows_box = run_test_box()
    rows_path = run_test_pathological()

    all_rows = rows_mpc + rows_box + rows_path
    summary = aggregate(all_rows)

    yaml_out = {
        "experiment": "projection_variants_v1",
        "timestamp": datetime.now().isoformat(),
        "variants": [v[0] for v in VARIANTS],
        "summary": summary,
        "raw": all_rows,
    }
    yaml_path = RESULTS_DIR / "benchmark_summary.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_out, f, default_flow_style=False, sort_keys=False)
    print(f"\nSaved: {yaml_path}")

    # Also save a numpy npz with raw arrays for plotting if needed
    npz = {}
    for variant in [v[0] for v in VARIANTS]:
        rel_errs = [r["rel_err"] for r in all_rows if r["name"] == variant]
        iters = [r["iters"] for r in all_rows if r["name"] == variant]
        wall = [r["wall_ms"] for r in all_rows if r["name"] == variant]
        npz[f"{variant}_rel_err"] = np.array(rel_errs, dtype=float)
        npz[f"{variant}_iters"] = np.array(iters, dtype=float)
        npz[f"{variant}_wall_ms"] = np.array(wall, dtype=float)
    npz_path = RESULTS_DIR / "benchmark_raw.npz"
    np.savez(npz_path, **npz)
    print(f"Saved: {npz_path}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()

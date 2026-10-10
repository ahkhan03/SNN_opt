"""Lean CPU baseline: equivalence to the v0.5 baseline and the library Dykstra.

Builds fpga/cpu_baseline once per session and compares it against
fpga/kv260_v05/src/cpu_baseline_v05.cpp (built unmodified apart from the
dimension cap, which a test-local copy lifts to the 1024 contract).
"""

from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
LEAN = ROOT / "fpga" / "cpu_baseline"
V05_SRC = ROOT / "fpga" / "kv260_v05" / "src"
# Optional directory of recorded band-fleet bundles (fleet_n<N>_it2000.bin);
# the synthetic band generator below is always exercised.
FLEET_DIR = Path(os.environ.get("CPU_BASELINE_FLEET_DIR", "/nonexistent"))
SLOW = os.environ.get("SNN_OPT_SLOW_TESTS") == "1"

pytestmark = pytest.mark.skipif(
    shutil.which("g++") is None, reason="g++ not available")

FLAGS = ["-O3", "-march=native", "-funroll-loops", "-std=c++17"]
# Bitwise comparison against the library-derived baseline needs one fixed
# floating-point contraction policy on both sides; the benchmark build keeps
# the compiler default.
EXACT = FLAGS + ["-ffp-contract=off"]


def _run(cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw)


@pytest.fixture(scope="session")
def binaries(tmp_path_factory):
    out = tmp_path_factory.mktemp("cpu_baseline_bins")
    # Benchmark build, through the shipped script.
    script = _run([str(LEAN / "build_cpu_baseline.sh"), str(out)])
    assert "cpu_baseline" in script.stdout
    # Reference: the v0.5 harness with its 64x64 guard lifted for the test.
    text = (V05_SRC / "cpu_baseline_v05.cpp").read_text()
    patched = text.replace("msrp_v05::load_problem(argv[1])",
                           "msrp_v05::load_problem(argv[1], false, 1024, 1024)")
    assert patched != text
    ref_src = out / "reference_v05.cpp"
    ref_src.write_text(patched)
    _run(["g++", *EXACT, f"-I{V05_SRC}", str(ref_src), "-o", str(out / "ref")])
    _run(["g++", *EXACT, "-fopenmp", str(LEAN / "cpu_baseline.cpp"),
          "-o", str(out / "lean_exact")])
    return {"bench": out / "cpu_baseline", "lean": out / "lean_exact",
            "ref": out / "ref"}


# ---- bundle I/O -----------------------------------------------------------
MAGIC = b"MSRPDL1\0"


def write_bundle(path, A, b, C, d, x0, k0, tol, lower, upper, iters, cap,
                 has_lower=True, has_upper=True):
    n, m = A.shape[0], C.shape[0]
    norms = (C * C).sum(axis=1)
    scale = np.where(norms > 0, 1.0 / np.sqrt(np.where(norms > 0, norms, 1)), 0.0)
    G = C @ C.T
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<7I", 1, n, m, iters, cap,
                            int(has_lower), int(has_upper)))
        f.write(struct.pack("<4d", k0, tol, lower, upper))
        for a in (A, b, C, d, norms, scale, G, x0):
            f.write(np.ascontiguousarray(a, dtype="<f8").tobytes())


def read_bundle(path):
    raw = Path(path).read_bytes()
    assert raw[:8] == MAGIC
    _, n, m, iters, cap, hl, hu = struct.unpack_from("<7I", raw, 8)
    k0, tol, lower, upper = struct.unpack_from("<4d", raw, 36)
    arr = np.frombuffer(raw, dtype="<f8", offset=68)
    pos = 0

    def take(count, shape):
        nonlocal pos
        v = arr[pos:pos + count].reshape(shape).copy()
        pos += count
        return v

    A = take(n * n, (n, n))
    b = take(n, (n,))
    C = take(m * n, (m, n))
    d = take(m, (m,))
    pos += 2 * m + m * m
    x0 = take(n, (n,))
    return dict(n=n, m=m, iters=iters, cap=cap, has_lower=bool(hl),
                has_upper=bool(hu), k0=k0, tol=tol, lower=lower, upper=upper,
                A=A, b=b, C=C, d=d, x0=x0)


def random_problem(path, seed, n, m, iters=300, upper=True):
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, n))
    A = M @ M.T / n + 0.2 * np.eye(n)
    k0 = 0.9 / np.linalg.eigvalsh(A).max()
    b = rng.standard_normal(n) * 2.0
    C = rng.standard_normal((m, n))
    d = rng.standard_normal(m) * 0.5 - 0.5 * np.abs(C).sum(axis=1) * 0.1
    x0 = rng.uniform(-0.5, 1.5, n)
    write_bundle(path, A, b, C, d, x0, k0, 1e-9, 0.0, 1.0, iters, 5000,
                 has_lower=True, has_upper=upper)


def band_problem(path, n, iters=2000):
    """Economic-dispatch-shaped problem: a two-row band on sum(x), box [0, 1]."""
    rng = np.random.default_rng(1000 + n)
    M = rng.standard_normal((n, n))
    A = M @ M.T / n + 0.5 * np.eye(n)
    k0 = 0.9 / np.linalg.eigvalsh(A).max()
    b = -rng.uniform(0.2, 1.0, n)
    C = np.vstack([np.ones(n), -np.ones(n)])
    d = np.array([-0.45 * n, 0.35 * n])        # 0.35 n <= sum(x) <= 0.45 n
    x0 = np.full(n, 0.4)
    write_bundle(path, A, b, C, d, x0, k0, 1e-9, 0.0, 1.0, iters, 5000)


def run(binary, bundle, out, threads=1, mode=None, reps=1):
    cmd = [str(binary), str(bundle), str(out), "--warmups", "1",
           "--reps", str(reps)]
    if mode:
        cmd += ["--mode", mode]
    res = _run(cmd, env={**os.environ, "OMP_NUM_THREADS": str(threads)})
    return json.loads(res.stdout)


def read_output(path, n):
    raw = Path(path).read_bytes()
    x = np.frombuffer(raw, dtype="<f8", count=n, offset=16).copy()
    tele = np.frombuffer(raw, dtype="<u8", offset=16 + 8 * n)
    return x, tele


# ---- canonical mode vs the v0.5 harness ----------------------------------
def _assert_same_as_reference(binaries, bundle, tmp_path):
    ref_json = run(binaries["ref"], bundle, tmp_path / "ref.bin")
    new_json = run(binaries["lean"], bundle, tmp_path / "new.bin")
    assert ref_json["status"] == new_json["status"]
    assert ref_json["events"] == new_json["events"] > 0
    assert ref_json["digest"] == new_json["digest"]
    # Whole output file: x bitwise plus every telemetry word (row, lower,
    # upper, total, cap rechecks, digest, first/last candidate, threads).
    assert (tmp_path / "ref.bin").read_bytes() == (tmp_path / "new.bin").read_bytes()
    return new_json


def _band_bundle(tmp_path, n, recorded):
    """A recorded fleet bundle when available, else a synthetic band."""
    if recorded:
        bundle = FLEET_DIR / f"fleet_n{n}_it2000.bin"
        if not bundle.exists():
            pytest.skip(f"recorded fleet bundle not available: {bundle}")
        if n > 40 and not SLOW:
            raw = bytearray(bundle.read_bytes())
            struct.pack_into("<I", raw, 8 + 4 * 3, 200)
            bundle = tmp_path / "short.bin"
            bundle.write_bytes(raw)
        return bundle
    bundle = tmp_path / "band.bin"
    band_problem(bundle, n, 2000 if n <= 40 or SLOW else 200)
    return bundle


@pytest.mark.parametrize("recorded", [False, True], ids=["synthetic", "recorded"])
@pytest.mark.parametrize("n", [10, 40, 160, 320])
def test_canonical_matches_v05_on_band_fleet(binaries, tmp_path, n, recorded):
    bundle = _band_bundle(tmp_path, n, recorded)
    out = _assert_same_as_reference(binaries, bundle, tmp_path)
    assert out["total"] == out["rows"] + out["lower"] + out["upper"]


@pytest.mark.parametrize("seed,n,m,upper", [
    (1, 12, 3, False), (2, 30, 5, True), (3, 64, 9, True),
    (4, 100, 2, True), (5, 37, 17, False)])
def test_canonical_matches_v05_on_random_problems(binaries, tmp_path, seed, n, m, upper):
    bundle = tmp_path / "p.bin"
    random_problem(bundle, seed, n, m, upper=upper)
    out = _assert_same_as_reference(binaries, bundle, tmp_path)
    assert out["rows"] > 0 and out["lower"] > 0
    if upper:
        assert out["upper"] > 0


def test_default_contraction_stays_close_to_reference(binaries, tmp_path):
    """The benchmark build lets the compiler contract to FMA; the event
    sequence is unchanged and x moves by rounding only."""
    bundle = tmp_path / "p.bin"
    random_problem(bundle, 7, 48, 6)
    ref = run(binaries["ref"], bundle, tmp_path / "ref.bin")
    new = run(binaries["bench"], bundle, tmp_path / "new.bin")
    assert ref["events"] == new["events"] and ref["digest"] == new["digest"]
    xr, _ = read_output(tmp_path / "ref.bin", 48)
    xn, _ = read_output(tmp_path / "new.bin", 48)
    assert np.max(np.abs(xr - xn)) < 1e-9


# ---- threads --------------------------------------------------------------
@pytest.mark.parametrize("mode", ["canonical", "dykstra"])
@pytest.mark.parametrize("n,m", [(10, 2), (97, 4), (160, 2)])
def test_four_threads_bitwise_equal_one_thread(binaries, tmp_path, mode, n, m):
    bundle = tmp_path / "p.bin"
    random_problem(bundle, 11 + n, n, m, iters=150)
    one = run(binaries["bench"], bundle, tmp_path / "one.bin", 1, mode)
    four = run(binaries["bench"], bundle, tmp_path / "four.bin", 4, mode)
    assert one["threads"] == 1 and four["threads"] == 4
    x1, t1 = read_output(tmp_path / "one.bin", n)
    x4, t4 = read_output(tmp_path / "four.bin", n)
    assert x1.tobytes() == x4.tobytes()
    keep = [i for i in range(len(t1)) if i != 13]  # word 13 is the thread count
    assert t1[keep].tolist() == t4[keep].tolist()
    for key in ("rows", "lower", "upper", "rounds", "row_steps"):
        assert one.get(key) == four.get(key)


# ---- dykstra mode vs the library -----------------------------------------
def _library_dykstra(bundle_path, iters):
    pytest.importorskip("snn_opt._kernel",
                        reason="compiled library kernel not built")
    from snn_opt import (
        ConvergenceConfig,
        OptimizationProblem,
        SNNSolver,
        SolverConfig,
        box_projector,
        joint_dykstra_projector,
    )
    p = read_bundle(bundle_path)
    n = p["n"]
    cand = joint_dykstra_projector(
        p["C"], p["d"], members=(box_projector(p["lower"], p["upper"]),))
    prob = OptimizationProblem(p["A"], p["b"], np.zeros((0, n)), np.zeros(0),
                               nonlinear_candidates=(cand,))
    cfg = SolverConfig(k0=p["k0"], max_iterations=iters, backend="c_serial",
                       record_trajectory=False, record_spike_history=False,
                       convergence=ConvergenceConfig(enable_early_stopping=False))
    res = SNNSolver(prob, cfg).solve(p["x0"])
    rounds = np.asarray(res.dykstra_inner_iterations_per_step)
    return np.asarray(res.final_x), float(rounds.mean()), res.dykstra_inner_cap_hits


@pytest.mark.parametrize("recorded", [False, True], ids=["synthetic", "recorded"])
@pytest.mark.parametrize("n", [10, 40, 160])
def test_dykstra_matches_library_on_band_fleet(binaries, tmp_path, n, recorded):
    bundle = _band_bundle(tmp_path, n, recorded)
    iters = int(struct.unpack_from("<I", bundle.read_bytes(), 8 + 4 * 3)[0])
    xl, rounds_lib, cap_hits = _library_dykstra(bundle, iters)
    got = run(binaries["bench"], bundle, tmp_path / "o.bin", 1, "dykstra")
    x, _ = read_output(tmp_path / "o.bin", n)
    assert cap_hits == 0 and got["guard_hits"] == 0 and got["round_caps"] == 0
    assert np.max(np.abs(x - xl)) < 1e-9
    # The library and this harness reach the same point but count rounds with
    # different stop-test bookkeeping (measured gap: at most about 2 rounds per
    # iteration, up to 6% on the synthetic n=40 band).
    assert abs(got["rounds"] / iters - rounds_lib) < 0.08 * rounds_lib + 0.05
    assert got["row_steps"] > 0

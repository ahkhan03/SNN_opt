"""Executable feature x backend capability matrix.

Every software cell is a tiny strictly convex problem that :func:`probe`
solves through the public API and classifies from behaviour: ``native``
(solves, and on a compiled backend agrees with the Python backend),
``fallback`` (solves through a rewrite the code exposes), or ``raises``
(fails closed with a ``ValueError``). Each row declares the classification it
expects, so the matrix in ``docs/capabilities.md`` cannot claim a cell that
the code does not deliver. ``tools/render_capabilities.py`` renders it and
``tests/test_capabilities.py`` checks expectations and freshness.

FPGA packages cannot be executed here; their boundaries are declared once in
:data:`FPGA_BOUNDARIES`, each entry citing the files that state it.
"""

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import scipy.sparse as sp

import snn_opt as so
from snn_opt.solver import ConvergenceConfig
from snn_opt.transforms import EigenbasisTransform

# (strategy, backend) column order of the software matrix.
COLUMNS = (
    ("greedy", "python"), ("greedy", "c_serial"), ("greedy", "c_openmp"),
    ("dykstra", "python"), ("dykstra", "c_serial"),
    ("eigenbasis", "python"), ("eigenbasis", "c_serial"),
    ("ivp", "python"), ("fixed", "python"),
)
CODES = {"N": "native", "F": "fallback", "R": "raises", "-": "n/a"}
PARITY_ATOL = 1e-6


@dataclass(frozen=True)
class Setup:
    """One set fixture: ``min 1/2 x'Ax + b'x`` s.t. rows, candidates, bounds."""
    A: np.ndarray
    b: np.ndarray
    C: object
    d: np.ndarray
    candidates: tuple = ()
    lower: Optional[float] = None
    upper: Optional[float] = None


@dataclass(frozen=True)
class Row:
    set_type: str
    build: Callable[[], Setup]
    expect: str  # one code from CODES per COLUMNS entry, space separated

    def expected(self, column: int) -> str:
        return CODES[self.expect.split()[column]]


@dataclass(frozen=True)
class Outcome:
    status: str  # native | fallback | raises | n/a | error
    detail: str = ""


def _qp(n, rows=0, seed=0):
    rng = np.random.default_rng(seed)
    A = np.diag(rng.uniform(1.0, 3.0, n))
    C = rng.normal(size=(rows, n))
    return A, rng.normal(size=n), C, -rng.uniform(0.1, 0.5, rows)


def _with(n, rows=0, candidates=(), lower=None, upper=None, sparse=False):
    A, b, C, d = _qp(n, rows)
    return Setup(A, b, sp.csr_matrix(C) if sparse else C, d, tuple(candidates), lower, upper)


def _equality(pair_of_rows):
    A, b, _, _ = _qp(6)
    B = np.random.default_rng(1).normal(size=(2, 6))
    h = np.array([0.3, -0.2])
    if pair_of_rows:
        return Setup(A, b, np.vstack([B, -B]), np.concatenate([-h, h]))
    return Setup(A, b, np.zeros((0, 6)), np.zeros(0), (so.AffineSubspaceProjector(B, h),))


def _lift(fn):
    rng = np.random.default_rng(2)
    A, b, _, _ = _qp(4)
    p = fn(A, b, rng.normal(size=(3, 4)), np.zeros(3), np.array([1.0, 0, 0, 0]), 2.0).problem
    return Setup(p.A, p.b, p.C, p.d, tuple(p.nonlinear_candidates))


def _user_cutter():
    return so.CutterCandidate(value=lambda x: float(x @ x) - 1.0, jacobian=lambda x: 2.0 * x,
                              name="user_ball")


def _clip(x):
    return np.clip(x, -1.0, 1.0)


R = range(1, 6)
ROWS = [
    #     columns: G py, G c_serial, G c_openmp, D py, D c_serial, eig py, eig c_serial, IVP, FIX
    Row("linear rows, dense", lambda: _with(6, 3), "N N N N N N N N N"),
    Row("linear rows, dense, m > 4096", lambda: _with(6, 4097), "F R R - - - - - -"),
    Row("linear rows, scipy sparse", lambda: _with(6, 3, sparse=True), "F R R R R R R N N"),
    Row("scalar box only (m=0)", lambda: _with(6, lower=-1.0, upper=1.0), "N N N F R F F N R"),
    Row("scalar box + rows", lambda: _with(6, 3, lower=-1.0, upper=1.0), "N N N F R F F N R"),
    Row("equality as opposed rows", lambda: _equality(True), "N N N N N N N N N"),
    Row("AffineSubspaceProjector", lambda: _equality(False), "N N N N N R R R R"),
    Row("halfspace_projector", lambda: _with(6, candidates=[so.halfspace_projector(np.ones(6), -0.5)]),
        "N N N N N R R R R"),
    Row("ball_projector", lambda: _with(6, candidates=[so.ball_projector(range(6), 0.5)]),
        "N N N N N R R R R"),
    Row("soc_projector", lambda: _with(6, candidates=[so.soc_projector(0, R)]), "N N N N N R R R R"),
    Row("scaled_soc_projector", lambda: _with(6, candidates=[so.scaled_soc_projector(0, R, 0.5)]),
        "N N N N N R R R R"),
    Row("psd_cone_projector 3x3", lambda: _with(6, candidates=[so.psd_cone_projector(3)]),
        "N N N N N R R R R"),
    Row("psd_cone_projector 9x9", lambda: _with(45, candidates=[so.psd_cone_projector(9)]),
        "N R R N R R R R R"),
    Row("spectral_ball_projector 3x3", lambda: _with(9, candidates=[so.spectral_ball_projector(3)]),
        "N N N N N R R R R"),
    Row("spectral_ball_projector 9x9", lambda: _with(81, candidates=[so.spectral_ball_projector(9)]),
        "N R R N R R R R R"),
    Row("spectral_ball_cutter 3x3", lambda: _with(9, candidates=[so.spectral_ball_cutter(3)]),
        "N N N R R R R R R"),
    Row("spectral_ball_cutter 9x9", lambda: _with(81, candidates=[so.spectral_ball_cutter(9)]),
        "N R R R R R R R R"),
    Row("lift_soc_l1", lambda: _lift(so.lift_soc_l1), "N N N N N R R R R"),
    Row("lift_soc_l2", lambda: _lift(so.lift_soc_l2), "N N N N N R R R R"),
    Row("nested DykstraProjector", lambda: _with(6, candidates=[so.DykstraProjector(
        [so.joint_dykstra_projector(*_qp(6, 3)[2:]), so.ball_projector(range(6), 0.5)])]),
        "N R R N R R R R R"),
    Row("user ProjectorCandidate", lambda: _with(6, candidates=[so.ProjectorCandidate(_clip)]),
        "N R R N R R R R R"),
    Row("user CutterCandidate", lambda: _with(6, candidates=[_user_cutter()]), "N R R R R R R R R"),
]
if hasattr(so, "box_projector"):  # the built-in box set, when this build has it
    ROWS.append(Row("box_projector (built-in)",
                    lambda: _with(6, 3, candidates=[so.box_projector(-1.0, 1.0)]),
                    "N N N N N R R R R"))


def _problem(setup, strategy):
    if strategy != "dykstra":
        return so.OptimizationProblem(setup.A, setup.b, setup.C, setup.d,
                                      nonlinear_candidates=setup.candidates)
    members = list(setup.candidates)
    if setup.lower is not None or setup.upper is not None:
        members.append(_clip)  # the only way to put a box in Dykstra without a built-in
    cand = (so.joint_dykstra_projector(setup.C, setup.d, members=members) if setup.C.shape[0]
            else so.DykstraProjector(members))
    n = setup.b.size
    return so.OptimizationProblem(setup.A, setup.b, np.zeros((0, n)), np.zeros(0),
                                  nonlinear_candidates=(cand,))


def _config(setup, strategy, backend):
    kw = dict(backend=backend, record_trajectory=backend == "python", max_iterations=30,
              convergence=ConvergenceConfig(enable_early_stopping=False))
    if strategy != "dykstra":
        kw.update(lower_bound=setup.lower, upper_bound=setup.upper)
    if strategy == "eigenbasis":
        kw["transform"] = "eigenbasis"
    elif strategy == "ivp":
        kw.update(integration_method="ivp", t_end=0.5)
    elif strategy == "fixed":
        kw["projection_method"] = "fixed"
    return so.SolverConfig(**kw)


def _solve(setup, strategy, backend):
    """Return (SolverResult, fallback_tag) or raise."""
    problem = _problem(setup, strategy)
    config = _config(setup, strategy, backend)
    x0 = np.random.default_rng(3).normal(size=setup.b.size)
    solver = so.SNNSolver(problem, config)
    tag = ""
    if strategy == "eigenbasis":
        # the precise applicability check must fire before any rewrite is attempted
        EigenbasisTransform().check_applicable(problem, config)
    if strategy == "dykstra" and (setup.lower is not None or setup.upper is not None):
        tag = "box as a callable member"
    elif strategy == "eigenbasis" and EigenbasisTransform().forward(problem, x0,
                                                                    config).consumes_bounds:
        tag = "box as rotated rows"
    elif (backend == "python" and strategy in ("greedy", "eigenbasis")
          and problem.n_constraints and solver._c_gram is None):
        tag = "residual recompute"
    return solver.solve(x0), tag


def backend_available(backend: str) -> bool:
    """True when this build can run ``backend`` (``c_openmp`` needs OpenMP)."""
    if backend == "python":
        return True
    try:
        from snn_opt import _kernel
    except ImportError:
        return False
    return backend != "c_openmp" or bool(getattr(_kernel, "HAS_OPENMP", False))


def probe(row: Row, column: int) -> Outcome:
    """Execute one cell and classify it from behaviour."""
    strategy, backend = COLUMNS[column]
    if row.expected(column) == "n/a":
        return Outcome("n/a")
    setup = row.build()
    try:
        res, tag = _solve(setup, strategy, backend)
    except (ValueError, TypeError) as exc:  # fail-closed: the message names the setting
        name = "" if isinstance(exc, ValueError) else f"{type(exc).__name__}: "
        return Outcome("raises", name + " ".join(str(exc).split()))
    except Exception as exc:  # anything else is a defect, never a capability
        return Outcome("error", f"{type(exc).__name__}: {exc}")
    if backend != "python":
        try:
            ref, _ = _solve(setup, strategy, "python")
        except (ValueError, TypeError) as exc:
            return Outcome("error", f"no independent python reference: {type(exc).__name__}")
        gap = np.max(np.abs(ref.final_x - res.final_x))
        if gap > PARITY_ATOL:
            return Outcome("error", f"python/{backend} mismatch {gap:.2e}")
        if (len(ref.dykstra_inner_iterations_per_step)
                or len(res.dykstra_inner_iterations_per_step)):
            for attr in ("dykstra_inner_iterations_per_step",
                         "dykstra_inner_projection_events_per_step"):
                if not np.array_equal(getattr(ref, attr), getattr(res, attr)):
                    return Outcome("error", f"python/{backend} Dykstra schedule differs: {attr}")
    return Outcome("fallback", tag) if tag else Outcome("native")


# Declared FPGA boundaries. Values are native | software | not_qualified | absent:
# software = the package states the feature stays host-side; absent = the
# package does not provide it; not_qualified = present or unstated but not
# board-qualified. Each package cites the files that state its boundary.
FPGA_VALUES = ("native", "software", "not_qualified", "absent")
FPGA_FEATURES = ("rows", "scalar_lower_bound", "scalar_upper_bound", "box_only_m0",
                 "vector_bounds", "ball", "scaled_soc", "affine_subspace", "psd_spectral",
                 "soc_lifts", "dykstra", "callbacks", "eigenbasis_mode")
_V0607 = dict(rows="native", scalar_lower_bound="native", scalar_upper_bound="native",
              box_only_m0="not_qualified", vector_bounds="not_qualified",
              affine_subspace="absent", psd_spectral="absent", soc_lifts="absent",
              dykstra="absent", callbacks="absent", eigenbasis_mode="not_qualified")
FPGA_BOUNDARIES = {
    "fpga/kv260_v05": dict(
        evidence=("fpga/kv260_v05/README.md",),
        rows="native", scalar_lower_bound="native", scalar_upper_bound="not_qualified",
        box_only_m0="not_qualified", vector_bounds="not_qualified", ball="absent",
        scaled_soc="absent", affine_subspace="absent", psd_spectral="absent",
        soc_lifts="absent", dykstra="absent", callbacks="absent",
        eigenbasis_mode="not_qualified"),
    "fpga/kv260_v06": dict(
        _V0607, evidence=("fpga/kv260_v06/README.md",
                          "fpga/kv260_v06/evidence/board_2026-09-06_37f33fcfe2ca"),
        ball="absent", scaled_soc="absent"),
    "fpga/kv260_v07": dict(
        _V0607, evidence=("fpga/kv260_v07/README.md",
                          "fpga/kv260_v07/evidence/board_2026-09-24_93eebcad9f2f/README.md"),
        ball="native", scaled_soc="native", affine_subspace="software",
        psd_spectral="software", soc_lifts="software", dykstra="software",
        callbacks="software"),
}

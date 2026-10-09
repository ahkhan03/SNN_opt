"""Economic-dispatch band regression for native Dykstra over rows + box.

The fixture ``fixtures/ed_band_n40.npz`` is a snapshot of the n=40 band
problem from the economic-dispatch study (scaled ``A, b``, unit-normalised
band rows ``C, d``, ``k0``, the default start ``x0``) plus the affine map
``p = p_off + p_scale * x`` back to generator outputs and the reference
dispatch ``p_ref`` (per unit, ``s_base`` MW). The rows and the ``[0, 1]``
box form one Dykstra candidate; the box is a built-in member, so the
compiled backend runs the whole projection natively.

The study's reference error is 0.0024 MW at 20000 iterations and 0.0028 MW
at 1000. The default test runs 1000 native iterations plus a short
Python-vs-native parity run; the full 20000-iteration reproduction is
opt-in through ``SNN_OPT_SLOW_TESTS=1`` because it takes several seconds.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from snn_opt import (
    ConvergenceConfig,
    OptimizationProblem,
    SNNSolver,
    SolverConfig,
    box_projector,
    joint_dykstra_projector,
)

pytest.importorskip(
    "snn_opt._kernel",
    reason="compiled C++ kernel not built (python setup.py build_ext --inplace)",
)

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "ed_band_n40.npz"


def _fixture():
    with np.load(_FIXTURE) as data:
        return {key: data[key] for key in data.files}


def _solve(fx, backend, iterations):
    n = fx["b"].size
    candidate = joint_dykstra_projector(fx["C"], fx["d"], members=(box_projector(0.0, 1.0),))
    problem = OptimizationProblem(
        fx["A"], fx["b"], np.zeros((0, n)), np.zeros(0), nonlinear_candidates=(candidate,))
    config = SolverConfig(
        k0=float(fx["k0"]),
        max_iterations=iterations,
        backend=backend,
        record_trajectory=backend == "python",
        record_spike_history=False,
        convergence=ConvergenceConfig(enable_early_stopping=False),
    )
    return SNNSolver(problem, config).solve(fx["x0"])


def _dispatch_error_mw(fx, x):
    p = fx["p_off"] + fx["p_scale"] * np.asarray(x, dtype=float)
    return float(np.max(np.abs(p - fx["p_ref"])) * float(fx["s_base"]))


def _assert_feasible(fx, x):
    x = np.asarray(x, dtype=float)
    assert max(0.0, -x.min(), x.max() - 1.0) <= 1e-10
    assert float(np.max(fx["C"] @ x + fx["d"])) <= 1e-10


def test_ed_band_native_dykstra_accuracy_1k():
    fx = _fixture()
    result = _solve(fx, "c_serial", 1000)
    _assert_feasible(fx, result.final_x)
    assert _dispatch_error_mw(fx, result.final_x) <= 0.003
    # Every outer step that projects reaches the intersection, not the cap.
    assert result.dykstra_inner_cap_hits == 0


def test_ed_band_dykstra_python_native_parity():
    fx = _fixture()
    py = _solve(fx, "python", 200)
    native = _solve(fx, "c_serial", 200)
    assert float(np.max(np.abs(py.final_x - native.final_x))) <= 1e-9
    np.testing.assert_array_equal(
        native.dykstra_inner_iterations_per_step, py.dykstra_inner_iterations_per_step)
    np.testing.assert_array_equal(
        native.dykstra_inner_projection_events_per_step,
        py.dykstra_inner_projection_events_per_step)


@pytest.mark.skipif(os.environ.get("SNN_OPT_SLOW_TESTS") != "1",
                    reason="20000-iteration reproduction; set SNN_OPT_SLOW_TESTS=1")
def test_ed_band_native_dykstra_reference_20k():
    fx = _fixture()
    result = _solve(fx, "c_serial", 20000)
    _assert_feasible(fx, result.final_x)
    assert _dispatch_error_mw(fx, result.final_x) <= 0.0025

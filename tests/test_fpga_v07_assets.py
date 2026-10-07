"""Software-side contract checks for the KV260 v07 native-reset package."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "fpga" / "kv260_v07" / "src"
_spec = importlib.util.spec_from_file_location("v07_model", SRC / "kernel_model.py")
_model = importlib.util.module_from_spec(_spec)
assert _spec.loader
sys.modules["v07_model"] = _model
_spec.loader.exec_module(_model)


def test_cones_off_schedule_and_ball_soc_events():
    n = 6
    A = np.eye(n)
    b = np.zeros(n)
    cones = [_model.Cone("ball", 0, 3, radius=1.0), _model.Cone("scaled_soc", 3, 3, mu=0.4)]
    out = _model.solve_v07(
        A,
        b,
        x0=np.array([2.0, 0.0, 0.0, -1.0, 2.0, 0.0]),
        cones=cones,
        n_iters=2,
        projection_cap=16,
    )
    assert out["cone_count"] == 2
    assert any(e[1] == "cone" for e in out["events"])
    assert np.linalg.norm(out["x"][:3]) <= 1 + 2**-23
    assert np.linalg.norm(out["x"][4:]) <= 0.4 * out["x"][3] + 2**-23


def test_cone_rejections_are_explicit():
    with pytest.raises(ValueError, match="overflow"):
        _model.validate_cone_table([_model.Cone("ball", i, 1, 1) for i in range(65)], 100)
    with pytest.raises(ValueError, match="overlap"):
        _model.validate_cone_table([_model.Cone("ball", 0, 3, 1), _model.Cone("ball", 2, 2, 1)], 8)
    with pytest.raises(ValueError, match="inside"):
        _model.validate_cone_table([_model.Cone("ball", 7, 2, 1)], 8)
    with pytest.raises(ValueError, match="fit"):
        _model.validate_cone_table([_model.Cone("ball", 0, 3, 128)], 8)
    with pytest.raises(ValueError, match="positive|at least"):
        _model.validate_cone_table([_model.Cone("scaled_soc", 0, 3, mu=0)], 8)
    with pytest.raises(ValueError, match="contiguous"):
        _model.validate_contiguous_indices([1, 2, 4], 8)


def test_v07_source_keeps_fixed_formats_and_table_cap():
    source = (SRC / "snn_qp_v07_kernel.cpp").read_text()
    assert "MAX_CONES = 64" in (SRC / "v07_cone_table.hpp").read_text()
    assert "ap_ufixed<49,25" in source
    assert "snn_qp_v07(" in source
    assert "configured_cone_count" in source
    assert "reciprocal_root" in source
    assert "ERR_BAD_CONES" in source
    assert "snn_qp_v07_conic" not in source
    assert not (SRC / "snn_qp_v07_conic_kernel.cpp").exists()


def test_v07_cones_share_the_resident_build_and_mailbox():
    kernel = (SRC / "snn_qp_v07_kernel.cpp").read_text()
    native = (SRC / "native_conic_v07.cpp").read_text()
    build = (ROOT / "fpga" / "kv260_v07" / "build" / "build_xclbin.sh").read_text()
    assert '"snn_qp_v07_kernel.cpp"' in kernel or "snn_qp_v07_kernel.cpp" in build
    assert "configured_cone_count" in kernel and "scan_cones" in kernel
    assert "MAILBOX_SEQUENCE" in kernel and "MAILBOX_SEQUENCE" in native
    assert "snn_qp_v07_conic" not in build

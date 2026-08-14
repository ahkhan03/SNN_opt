"""Fast contract tests for the restricted KV260 v0.5 reference assets."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

# The frozen KV260 package's identity (pinned SHA-256 hashes, POSIX-relative
# manifest paths) is defined on POSIX checkouts: a Windows checkout rewrites
# line endings under autocrlf (changing every hash) and reports backslash
# paths. The qualification surface itself (Vitis/XRT/KV260) is Linux-only,
# so these identity checks are meaningless there.
pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="frozen FPGA package identity is defined on POSIX checkouts; "
    "the KV260 toolchain surface is Linux-only")

ROOT = Path(__file__).resolve().parents[1]
FPGA = ROOT / "fpga" / "kv260_v05"
SRC = FPGA / "src"
BUILD = FPGA / "build"

QUALIFIED_HASHES = {
    "src/snn_qp_v05_kernel.cpp": (
        "927e30945d07aae941f46655c3d8bfa0a7715ff5d704f1f926664505095d6ea1"
    ),
    "src/dt.h": (
        "d000c064e08f947621a093a506812d14064b9a3b2ded4e81263b0d538e43936a"
    ),
    "src/bench_support.hpp": (
        "e004afef8b88516671e6fe5d0b1b865996b84169a4ae433cdcd14219d5afd6f7"
    ),
    "src/cpu_baseline_v05.cpp": (
        "28524d4e029f7e132345a8847045ceeb8ffbb6e3c5c09eb3f76e1ae0d1a7082a"
    ),
    "src/host_kv260_v05.cpp": (
        "71064c323c7d1d353a973b940b599b935fab477ca003e83f1b4ea200608311fa"
    ),
    "src/msrp_bundle.hpp": (
        "ce1c262ac07ee75cc4a9eec9dd609b26841e3c16109aad8f4d5461e856dba682"
    ),
    "src/native_fixed_v05.cpp": (
        "9e0e66b422b8ccaf0769346d2276bfd4da401e75ec4c0e716161f8605bb3c986"
    ),
    "src/reference_v05.cpp": (
        "f1ebe1719b9d190b12b19cf0742fa7c5691d5678cab4df9e8d6741450f710fae"
    ),
    "src/v05_double_core.hpp": (
        "7b678665ea5157bafad7f0f20875762981cddc3f77d957151412558abf4a3d7e"
    ),
    "build/build_native.sh": (
        "c70ef4d82e375d3a6ae43532be1b149e6f95f1fce2e26a0982bb81419b56ac54"
    ),
    "build/build_xclbin.sh": (
        "1214de63b3058e31c2997b56b637558643cc53aa83cdc9bdd96195539ef60857"
    ),
    "build/connectivity.cfg": (
        "f1337abf4adf0a4104a2bdee6ffa4c7ed05d98c1f17284eb4e19a91e85c5736c"
    ),
    "build/msrp_v05.bif": (
        "4f4c6d5fcf077b91b680e2387a4120b03845e6dca5b2561fa55d6245ed1cf795"
    ),
    "build/msrp_v05.dts": (
        "cf41ce1e9599a0217967c619cb22db16d7d744b563570d9ee27740ba90f8da21"
    ),
    "build/run_hls.tcl": (
        "7f36db97992385fe63632701ed1f3b8bcce9e8d5ace49e2ab38ddca67f73e1e1"
    ),
    "env.yaml": (
        "abf324f7c4e0f20f6a4bbde829a31652fa01bbd7533db5203973d83e822f1dad"
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_imported_qualified_assets_preserve_exact_bytes() -> None:
    assert set(QUALIFIED_HASHES) == {
        str(path.relative_to(FPGA))
        for path in FPGA.rglob("*")
        if path.is_file()
        and path.name not in {"README.md", ".gitignore", "canonical_solve.hpp"}
    }
    for relative, expected in QUALIFIED_HASHES.items():
        assert _sha256(FPGA / relative) == expected


def test_kernel_contract_is_v05_faithful_and_binary64_at_boundary() -> None:
    source = (SRC / "snn_qp_v05_kernel.cpp").read_text(encoding="utf-8")
    dtype = (SRC / "dt.h").read_text(encoding="utf-8")
    assert "const double* A_in" in source
    assert "const double* C_in" in source
    assert "ap_fixed<DATA_W, DATA_I, AP_RND_CONV, AP_SAT>" in dtype
    assert "ap_fixed<ACC_W, ACC_I, AP_RND_CONV, AP_SAT>" in dtype
    assert "#define DATA_W 32" in dtype
    assert "#define DATA_I 8" in dtype
    assert "#define ACC_W 48" in dtype
    assert "#define ACC_I 16" in dtype

    residual = source.index("residual_rows:")
    rows = source.index("scan_rows:")
    lower = source.index("scan_lower:")
    upper = source.index("scan_upper:")
    cap = source.index("cap_residual_rows:")
    write = source.index("write_state:")
    assert residual < rows < lower < upper < cap < write
    assert "if (score > best)" in source
    assert "if (maximum > static_cast<acc_t>(ctol))" in source
    assert "status = 2;" in source
    assert "x[i] =" not in source[write:]
    assert "std::clamp" not in source


def test_bundle_and_capability_guards_are_explicit() -> None:
    bundle = (SRC / "msrp_bundle.hpp").read_text(encoding="utf-8")
    native = (SRC / "native_fixed_v05.cpp").read_text(encoding="utf-8")
    readme = (FPGA / "README.md").read_text(encoding="utf-8")
    assert "'M', 'S', 'R', 'P', 'D', 'L', '1'" in bundle
    assert "problem.n <= 0 || problem.n > 64 || problem.m <= 0" in bundle
    assert "problem.m > 64" in bundle
    assert "problem bundle contains trailing bytes" in bundle
    assert "v0.5 hardware path requires binary64 input" in native
    for boundary in (
        "box-only problems with `m == 0`",
        "vector-valued bounds",
        "early stopping or convergence behavior",
        "active upper-bound facets",
        "projection-budget-abort path",
        "range-violation telemetry",
    ):
        assert boundary in readme


def test_canonical_adapter_compiles_against_current_v05_core(
    tmp_path: Path,
) -> None:
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is not available")
    adapter = (SRC / "canonical_solve.hpp").read_text(encoding="utf-8")
    assert '#include "../../../src/snn_opt/_native/snn_qp_core.hpp"' in adapter
    assert _sha256(ROOT / "src/snn_opt/_native/snn_qp_core.hpp") == (
        "16704b2c646286e5c4ce5ee8ce023ba521bd889787a544df1eea772213900a4f"
    )
    unit = tmp_path / "adapter.cpp"
    unit.write_text(
        '#include "fpga/kv260_v05/src/canonical_solve.hpp"\n'
        "int main() { return 0; }\n",
        encoding="utf-8",
    )
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-fsyntax-only",
            "-I",
            str(ROOT),
            str(unit),
        ],
        check=True,
        cwd=ROOT,
    )


def test_build_recipe_keeps_qualified_platform_and_clock() -> None:
    env = (FPGA / "env.yaml").read_text(encoding="utf-8")
    xclbin = (BUILD / "build_xclbin.sh").read_text(encoding="utf-8")
    hls = (BUILD / "run_hls.tcl").read_text(encoding="utf-8")
    assert "device_part: xck26-sfvc784-2LV-c" in env
    assert "clock_mhz_requested: 200" in env
    assert "input_abi: binary64" in env
    assert 'clock_hz="200000000"' in xclbin
    assert "set_part {xck26-sfvc784-2LV-c}" in hls
    assert "create_clock -period 5.0" in hls

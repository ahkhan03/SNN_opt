"""Quantized k0-floor contract for the v09b host.

The integer chain is the schedule the host runs. These tests stay on a
workstation: no board, no network, and no synthesis. Three of them compare
against the shipped C++ kernel and therefore need the Vitis HLS headers:
set ``HLS_INCLUDE`` to a directory containing ``ap_fixed.h`` or they skip.
The gate is ``_hls_include_ready()`` below, documented in
``fpga/kv260_v09_bram/README.md`` and in ``build/build_native.sh``.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from fpga.kv260_v09_bram.tools.board.bundle import write_bundle
from fpga.kv260_v09_bram.tools.board.decay_choice import decide_for_bundle
from fpga.kv260_v09_bram.tools.board.decay_contract import (
    in_near_floor_band,
    level_count,
    load_contract,
    next_level,
)
from snn_opt import OptimizationProblem, SNNSolver, SolverConfig
from snn_opt.recommend import DecayRecommendation

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "fpga" / "kv260_v09_bram"
SRC = PKG / "src"

# Three-variable quadratic whose constant step stalls above 1e-4.
_STIFF_A = np.array([
    [0.11387419, 0.32740482, 0.38032890],
    [0.32740482, 3.10892627, 3.23929608],
    [0.38032890, 3.23929608, 3.95096198],
])
_STIFF_B = np.array([0.12279726, -0.81305707, 1.97858988])
_STIFF_C = np.array([
    [-0.44193274, -0.67092762, -0.59544234],
    [0.22153773, -0.88767343, 0.40367922],
    [0.13692966, 0.81618772, -0.56132689],
    [0.99483654, -0.10026541, -0.01571940],
    [-0.85639087, -0.27182262, 0.43898423],
    [-0.26361726, 0.25906587, -0.92918826],
])
_STIFF_D = np.array([
    -0.84749728, -0.43916840, -0.43455156,
    -0.58556949, -0.29355200, -0.32266863,
])

# The same probe already meets 1e-4, so decay is not indicated.
_HEALTHY_A = np.array([
    [3.70686090, 3.17021637, 1.62443914],
    [3.17021637, 5.30004987, 3.81985579],
    [1.62443914, 3.81985579, 4.36982807],
])
_HEALTHY_B = np.array([-0.03663783, -0.89724926, 0.23313208])
_HEALTHY_C = np.array([
    [-0.67444237, 0.34918978, 0.65053362],
    [-0.24708692, 0.44858512, 0.85890596],
])
_HEALTHY_D = np.array([-0.92114486, -0.37371861])

_QUALIFIED = (
    (50641056, 6291456, (50641056, 25320528, 12660264, 6291456)),
    (317556, 33554, (317556, 158778, 79389, 33554)),
)

_SCHEMA_KEYS = {
    "package": ("version", "commit"),
    "bundle": (
        "name", "path", "sha256", "n", "m", "iterations", "k0", "k0_quanta",
        "projection_cap", "constraint_tol", "has_lower", "has_upper",
        "start_mode",
    ),
    "contract": (
        "word_bits", "int_bits", "fraction_bits", "min_floor_quanta",
        "min_floor",
    ),
    "recommendation": (
        "floor", "segment", "levels", "reason", "meets_target", "evidence",
    ),
    "board": (
        "floor_requested", "floor_quanta", "floor_effective",
        "quantization_error", "clamped_to_min", "levels_capped",
        "chain_quanta", "chain", "segment", "segment_adjusted",
        "final_effective", "float_levels", "board_levels",
        "level_count_matches",
    ),
    "confirmation": (
        "kkt_normalized", "kkt_fit_status", "kkt_residual", "kkt_scale",
        "convergence_reason", "converged", "projection_budget_exhausted",
        "certificate_available", "meets_target",
    ),
}

_NO_DECAY = (
    "target_already_met",
    "residual_improves_with_iterations",
    "residual_not_linear_in_k0",
)
_REFUSE = (
    "min_floor_at_or_above_start",
    "probe_projection_budget_exhausted",
    "kkt_certificate_unavailable",
    "verification_projection_budget_exhausted",
    "no_usable_floor_within_budget",
)


def _require_gxx():
    if shutil.which("g++") is None:
        pytest.skip("g++ is absent")


def _macro(text, name):
    found = re.findall(
        rf"^[ \t]*#define[ \t]+{name}[ \t]+(\d+)[ \t]*$", text, re.MULTILINE)
    assert found, name
    return int(found[-1])


def _problem(A, b, C, d):
    return OptimizationProblem(
        A=np.asarray(A, float), b=np.asarray(b, float),
        C=np.asarray(C, float), d=np.asarray(d, float))


def _write_problem(path, problem, iterations):
    k0 = float(SNNSolver(
        problem, SolverConfig(max_iterations=iterations))._k0)
    write_bundle(
        path, problem.A, problem.b, problem.C, problem.d,
        np.zeros(problem.A.shape[0]), k0, 1e-6, 0.0, 0.0,
        iterations, 20000, has_lower=False, has_upper=False)
    return k0


def _assert_schema(document):
    assert document["schema"] == "snn-qp-decay-decision-v1"
    assert "decision" in document and "host_args" in document
    for section, keys in _SCHEMA_KEYS.items():
        assert section in document
        for key in keys:
            assert key in document[section]


def test_contract_matches_headers():
    contract = load_contract()
    env = (PKG / "env.yaml").read_text()
    dt = (SRC / "dt.h").read_text()
    header = (SRC / "v09b_decay.hpp").read_text()
    word_bits = _macro(dt, "DATA_W")
    int_bits = _macro(dt, "DATA_I")
    fixed = re.search(r"fixed_state:\s*ap_fixed<(\d+),(\d+),", env)
    assert fixed
    minima = re.findall(r"MIN_FLOOR_QUANTA\s*=\s*(\d+)", header)
    assert minima
    assert contract.word_bits == word_bits == int(fixed.group(1))
    assert contract.int_bits == int_bits == int(fixed.group(2))
    assert int(re.search(r"^word_bits:\s*(\d+)\s*$", env, re.M).group(1)) == word_bits
    assert int(re.search(r"^int_bits:\s*(\d+)\s*$", env, re.M).group(1)) == int_bits
    assert contract.min_floor_quanta == int(minima[0])
    assert int(re.search(r"^min_floor_quanta:\s*(\d+)\s*$", env, re.M).group(1)) == int(minima[0])
    assert contract.fraction_bits == word_bits - int_bits
    assert contract.quantum == 1.0 / float(1 << contract.fraction_bits)
    assert contract.min_floor == contract.dequantize(contract.min_floor_quanta)


def test_quantize_matches_cpp(tmp_path):
    _require_gxx()
    contract = load_contract()
    scale = float(1 << contract.fraction_bits)
    values = [
        0.0, -1.0, 1e-12, 1.0, 100.0, 1e9,
        128.0 / scale, 127.5 / scale, 128.5 / scale, 129.5 / scale,
        50641056 / scale, 6291456 / scale, 317556 / scale, 33554 / scale,
        (2**31) / scale,
    ]
    pairs = [(start, floor) for start, floor, _expected in _QUALIFIED]
    pairs += [(50641056, contract.min_floor_quanta), (129, 128), (192, 128)]
    lines = [
        '#include "v09b_decay.hpp"',
        "#include <cstdint>",
        "#include <cstdio>",
        "int main() {",
        "  const double values[] = {"
        + ", ".join(float(value).hex() for value in values) + "};",
        "  const int nvalues = sizeof(values) / sizeof(values[0]);",
        "  for (int i = 0; i < nvalues; ++i)",
        '    std::printf("Q %d %u\\n", i,'
        " snn_v09b_decay::quantize_positive(values[i]));",
        "  struct Pair { std::uint32_t start; std::uint32_t floor; };",
        "  const Pair pairs[] = {"
        + ", ".join(f"{{{start}u, {floor}u}}" for start, floor in pairs) + "};",
        "  const int npairs = sizeof(pairs) / sizeof(pairs[0]);",
        "  for (int i = 0; i < npairs; ++i) {",
        "    const std::uint32_t start = pairs[i].start;",
        "    const std::uint32_t floor = pairs[i].floor;",
        '    std::printf("L %u %u %d\\n", start, floor,'
        " snn_v09b_decay::level_count(start, floor));",
        '    std::printf("N %u %u %u\\n", start, floor,'
        " snn_v09b_decay::next_level(start, floor));",
        '    std::printf("C");',
        "    std::uint32_t level = start;",
        '    std::printf(" %u", level);',
        "    for (int guard = 0; level > floor && guard < 64; ++guard) {",
        "      const std::uint32_t nxt = snn_v09b_decay::next_level(level, floor);",
        "      if (nxt == level) break;",
        '      std::printf(" %u", nxt);',
        "      level = nxt;",
        "    }",
        '    std::printf("\\n");',
        "  }",
        "  return 0;",
        "}",
        "",
    ]
    source = tmp_path / "probe.cpp"
    binary = tmp_path / "probe"
    source.write_text("\n".join(lines))
    subprocess.run(
        ["g++", "-std=c++17", f"-I{SRC}", str(source), "-o", str(binary)],
        check=True, capture_output=True, text=True)
    completed = subprocess.run(
        [str(binary)], check=True, capture_output=True, text=True)
    quanta = {}
    counts = {}
    steps = {}
    chains = {}
    for line in completed.stdout.splitlines():
        kind, *rest = line.split()
        if kind == "Q":
            quanta[int(rest[0])] = int(rest[1])
        elif kind == "L":
            counts[(int(rest[0]), int(rest[1]))] = int(rest[2])
        elif kind == "N":
            steps[(int(rest[0]), int(rest[1]))] = int(rest[2])
        elif kind == "C":
            chain = tuple(int(item) for item in rest)
            chains[(chain[0], chain[-1])] = chain
    for index, value in enumerate(values):
        assert contract.quantize(value) == quanta[index]
    for start, floor in pairs:
        assert level_count(start, floor) == counts[(start, floor)]
        assert next_level(start, floor) == steps[(start, floor)]
        walked = chains[(start, floor)]
        if in_near_floor_band(start, floor):
            # level_count drops the start level the kernel still executes.
            assert level_count(start, floor) == 1
            assert walked == (start, floor)
            assert len(walked) != level_count(start, floor)
        else:
            assert contract.chain(start, floor) == walked
            assert len(walked) == level_count(start, floor)


def test_chain_qualified_examples(tmp_path, monkeypatch):
    contract = load_contract()
    for start, floor, expected in _QUALIFIED:
        got = contract.chain(start, floor)
        assert got == expected
        assert len(got) == level_count(start, floor)
        assert _walk(start, floor) == got
        assert not in_near_floor_band(start, floor)

    floor = contract.min_floor_quanta
    got = contract.chain(50641056, floor)
    assert got[0] == 50641056 and got[-1] == floor
    assert len(got) == level_count(50641056, floor)
    assert _walk(50641056, floor) == got
    assert not in_near_floor_band(50641056, floor)

    # 129 and 192 sit in the band against floor 128. level_count returns 1.
    # The kernel executes [start, floor]. That schedule is a refusal.
    for start in (129, 192):
        assert in_near_floor_band(start, 128)
        assert level_count(start, 128) == 1
        assert _walk(start, 128) == (start, 128)
        assert len(_walk(start, 128)) != level_count(start, 128)
        _assert_band_refusal(tmp_path, monkeypatch, contract, start, 128)


def _walk(start, floor):
    levels = [start]
    level = start
    while level > floor:
        nxt = next_level(level, floor)
        if nxt == level:
            break
        levels.append(nxt)
        level = nxt
    return tuple(levels)


def test_decision_on_stiff_bundle(tmp_path):
    bundle = tmp_path / "stiff.bin"
    _write_problem(bundle, _problem(_STIFF_A, _STIFF_B, _STIFF_C, _STIFF_D), 8000)
    document, code = decide_for_bundle(bundle, target=1e-4)
    assert code == 0
    assert document["decision"] == "decay"
    _assert_schema(document)
    contract = load_contract()
    start_q = document["bundle"]["k0_quanta"]
    floor_q = document["board"]["floor_quanta"]
    chain = document["board"]["chain_quanta"]
    segment = document["board"]["segment"]
    iterations = document["bundle"]["iterations"]
    assert floor_q >= contract.min_floor_quanta
    assert floor_q < start_q
    assert not in_near_floor_band(start_q, floor_q)
    assert document["confirmation"]["meets_target"] is True
    assert chain[0] == start_q and chain[-1] == floor_q
    assert segment >= 1 and segment * len(chain) <= iterations
    floor_eff = document["board"]["floor_effective"]
    assert document["host_args"] == [
        "--k0-floor", repr(floor_eff), "--decay-segment", str(segment)]
    assert float(document["host_args"][1]) == floor_eff
    assert document["bundle"]["sha256"]
    assert document["bundle"]["name"] == bundle.name
    # Outside the repository the recorded path is the basename, never absolute.
    assert document["bundle"]["path"] == bundle.name
    assert not str(document["bundle"]["path"]).startswith("/")
    assert str(bundle.resolve()) not in json.dumps(document)
    assert document["bundle"]["start_mode"] == "bundle"


def test_no_decay_on_easy_bundle(tmp_path):
    bundle = tmp_path / "healthy.bin"
    _write_problem(
        bundle, _problem(_HEALTHY_A, _HEALTHY_B, _HEALTHY_C, _HEALTHY_D), 2000)
    document, code = decide_for_bundle(bundle, target=1e-4)
    assert code == 3
    assert document["decision"] == "no_decay"
    assert document["recommendation"]["reason"] == "target_already_met"
    assert document["recommendation"]["reason"] in _NO_DECAY
    assert document["host_args"] == []
    assert document["recommendation"]["floor"] is None


def test_clamp_flagged(tmp_path):
    bundle = tmp_path / "stiff.bin"
    _write_problem(bundle, _problem(_STIFF_A, _STIFF_B, _STIFF_C, _STIFF_D), 8000)
    # The model floor for this target sits below the fixed-point minimum.
    # A clamp is recorded. The confirmation at that floor still gates:
    # missing the target is a refusal, not a second flag.
    document, code = decide_for_bundle(bundle, target=1e-6)
    contract = load_contract()
    board = document["board"]
    assert board["clamped_to_min"] is True
    assert document["recommendation"]["evidence"]["floor_clamped_to_min"] is True
    assert board["floor_quanta"] == contract.min_floor_quanta
    assert board["floor_effective"] == contract.min_floor
    # Measured: the confirmation at the clamped floor misses 1e-6. The clamp
    # is recorded, the decision refuses, and the override is the only way.
    assert document["confirmation"]["meets_target"] is False
    assert document["decision"] == "refuse"
    assert code == 3
    assert document["host_args"] == []
    allowed, allowed_code = decide_for_bundle(
        bundle, target=1e-6, allow_above_target=True)
    assert allowed["decision"] == "decay"
    assert allowed_code == 0
    assert "clamped" not in allowed["decision"]


def _recommendation(floor, reason, meets=True, clamped=False):
    return DecayRecommendation(
        floor=floor,
        segment=None if floor is None else 10,
        levels=[] if floor is None else [floor],
        evidence={
            "meets_target": meets,
            "floor_clamped_to_min": clamped,
            "levels_capped": False,
        },
        reason=reason,
    )


def _patch_recommend(monkeypatch, recommendation):
    monkeypatch.setattr(
        "fpga.kv260_v09_bram.tools.board.decay_choice.recommend_k0_decay",
        lambda *args, **kwargs: recommendation,
    )


def _assert_band_refusal(tmp_path, monkeypatch, contract, start_q, floor_q):
    bundle = tmp_path / f"band_{start_q}.bin"
    _write_with_k0(bundle, contract.dequantize(start_q), iterations=64)
    _patch_recommend(
        monkeypatch, _recommendation(contract.dequantize(floor_q), "recommended"))
    document, code = decide_for_bundle(bundle, target=1e-4)
    board = document["board"]
    requested = contract.dequantize(floor_q)
    assert code == 3
    assert document["decision"] == "refuse"
    assert document["recommendation"]["reason"] == "floor_too_close_to_start"
    assert document["host_args"] == []
    # The refusal is before the chain is built. Known quanta stay filled.
    assert board["floor_requested"] == requested
    assert board["floor_quanta"] == floor_q
    assert board["floor_effective"] == requested
    assert board["quantization_error"] == 0.0
    assert board["chain_quanta"] == []
    assert board["chain"] == []


def _write_with_k0(path, k0, iterations):
    problem = _problem(np.eye(2), np.zeros(2), np.zeros((1, 2)), np.zeros(1))
    write_bundle(
        path, problem.A, problem.b, problem.C, problem.d,
        np.zeros(2), float(k0), 1e-6, 0.0, 0.0,
        iterations, 20000, has_lower=False, has_upper=False)


def _stiff_decision(tmp_path):
    bundle = tmp_path / "stiff.bin"
    _write_problem(bundle, _problem(_STIFF_A, _STIFF_B, _STIFF_C, _STIFF_D), 8000)
    document, code = decide_for_bundle(bundle, target=1e-4)
    assert code == 0
    return bundle, document


def _hls_include_ready():
    include = os.environ.get("HLS_INCLUDE", "")
    return bool(include) and (Path(include) / "ap_fixed.h").is_file()


@pytest.fixture(scope="module")
def mock_host(tmp_path_factory):
    _require_gxx()
    bindir = tmp_path_factory.mktemp("mock-host")
    built = subprocess.run(
        [str(PKG / "build" / "build_host.sh"), str(bindir)],
        check=True, capture_output=True, text=True)
    assert "mock" in built.stdout.lower()
    return bindir / "host_kv260_v09b"


@pytest.fixture(scope="module")
def native_bin(tmp_path_factory):
    _require_gxx()
    if not _hls_include_ready():
        pytest.skip("HLS_INCLUDE is unset or has no ap_fixed.h")
    bindir = tmp_path_factory.mktemp("native")
    built = subprocess.run(
        [str(PKG / "build" / "build_native.sh"), str(bindir)],
        check=False, capture_output=True, text=True)
    assert built.returncode == 0, built.stderr
    return bindir / "native_fixed_v09b"


def _run_host(host, bundle, out, host_json, args):
    completed = subprocess.run(
        [str(host), "mock.xclbin", str(bundle), str(out),
         "--one-shot", "--warmups", "0", "--reps", "1",
         "--json-out", str(host_json), *args],
        check=False, capture_output=True, text=True)
    return completed


def test_decision_labels(tmp_path, monkeypatch):
    bundle = tmp_path / "labels.bin"
    _write_with_k0(bundle, 0.05, iterations=32)
    for reason in _NO_DECAY:
        _patch_recommend(monkeypatch, _recommendation(None, reason, meets=False))
        document, code = decide_for_bundle(bundle, target=1e-4)
        assert code == 3
        assert document["decision"] == "no_decay"
        assert document["recommendation"]["reason"] == reason
        assert document["host_args"] == []
    for reason in _REFUSE:
        _patch_recommend(monkeypatch, _recommendation(None, reason, meets=False))
        document, code = decide_for_bundle(bundle, target=1e-4)
        assert code == 3
        assert document["decision"] == "refuse"
        assert document["host_args"] == []


def test_confirmation_gates_the_decision(tmp_path, monkeypatch):
    contract = load_contract()
    bundle = tmp_path / "gate.bin"
    start = contract.dequantize(50641056)
    floor = contract.dequantize(6291456)
    _write_with_k0(bundle, start, iterations=2000)
    meeting = _recommendation(floor, "recommended", meets=True)
    _patch_recommend(monkeypatch, meeting)

    class Result:
        def __init__(self, kkt):
            self.kkt_normalized = kkt
            self.kkt_fit_status = state["fit"]
            self.kkt_residual = kkt
            self.kkt_scale = 1.0
            self.converged = kkt <= 1e-4
            self.convergence_reason = (
                "kkt_target" if self.converged else "max_iterations")
            self.projection_budget_exhausted = state["aborted"]
            self.decay_levels = [start, floor]

    seen = {}
    state = {"kkt": 1e-3, "aborted": False, "fit": "ok"}

    class RecordingSolver:
        def __init__(self, problem, config):
            seen["config"] = config

        def solve(self, x, verbose=False):
            return Result(state["kkt"])

    monkeypatch.setattr(
        "fpga.kv260_v09_bram.tools.board.decay_choice.SNNSolver",
        RecordingSolver,
    )

    # The recommendation met the target. The confirmation missed it.
    document, code = decide_for_bundle(bundle, target=1e-4)
    assert code == 3
    assert document["decision"] == "refuse"
    assert document["recommendation"]["meets_target"] is True
    assert document["confirmation"]["meets_target"] is False
    assert document["confirmation"]["converged"] is False
    assert document["host_args"] == []
    assert seen["config"].convergence.enable_early_stopping is False
    assert seen["config"].k0 == start
    assert seen["config"].k0_decay_floor == floor

    document, code = decide_for_bundle(
        bundle, target=1e-4, allow_above_target=True)
    assert code == 0
    assert document["decision"] == "decay"
    assert document["recommendation"]["meets_target"] is True
    assert document["confirmation"]["meets_target"] is False
    assert document["host_args"]
    assert seen["config"].convergence.enable_early_stopping is False

    # The probe flag says the target was missed. The confirmation met it,
    # so the gate follows the confirmation and not the recommendation.
    _patch_recommend(
        monkeypatch,
        _recommendation(floor, "recommended_above_target", meets=False))
    state["kkt"] = 1e-6
    document, code = decide_for_bundle(bundle, target=1e-4)
    assert code == 0
    assert document["decision"] == "decay"
    assert document["recommendation"]["meets_target"] is False
    assert document["confirmation"]["meets_target"] is True
    assert document["host_args"]
    assert seen["config"].convergence.enable_early_stopping is False

    # An aborted sweep or an unavailable certificate is a miss even when the
    # number that accompanies it is tiny. The override is the only way through.
    _patch_recommend(monkeypatch, meeting)
    for patch in ({"aborted": True}, {"fit": "fit_failed"}):
        state.update({"kkt": 1e-9, "aborted": False, "fit": "ok"})
        state.update(patch)
        document, code = decide_for_bundle(bundle, target=1e-4)
        assert code == 3
        assert document["decision"] == "refuse"
        assert document["confirmation"]["meets_target"] is False
        assert document["host_args"] == []
        allowed, allowed_code = decide_for_bundle(
            bundle, target=1e-4, allow_above_target=True)
        assert allowed_code == 0
        assert allowed["decision"] == "decay"

    # An unset confirmation is the same miss.
    monkeypatch.setattr(
        "fpga.kv260_v09_bram.tools.board.decay_choice._confirm",
        lambda *args, **kwargs: None,
    )
    _patch_recommend(monkeypatch, meeting)
    document, code = decide_for_bundle(bundle, target=1e-4)
    assert code == 3
    assert document["decision"] == "refuse"
    assert document["recommendation"]["meets_target"] is True
    assert document["confirmation"]["meets_target"] is None
    assert document["host_args"] == []
    document, code = decide_for_bundle(
        bundle, target=1e-4, allow_above_target=True)
    assert code == 0
    assert document["decision"] == "decay"
    assert document["confirmation"]["meets_target"] is None


def test_decision_path_stays_inside_the_repo(monkeypatch):
    # A tracked bundle: the v09b fixtures are gitignored, so a candidate
    # worktree would not carry them and this test would only pass in a
    # developer checkout.
    bundle = (
        ROOT / "fpga" / "kv260_v06" / "evidence"
        / "board_2026-09-06_37f33fcfe2ca" / "fixtures" / "prob_6x18.bin"
    ).resolve()
    _patch_recommend(
        monkeypatch, _recommendation(None, "residual_improves_with_iterations", meets=False))
    document, code = decide_for_bundle(bundle, target=1e-4)
    assert code == 3
    assert document["decision"] == "no_decay"
    assert document["bundle"]["name"] == "prob_6x18.bin"
    relative = (
        "fpga/kv260_v06/evidence/board_2026-09-06_37f33fcfe2ca/"
        "fixtures/prob_6x18.bin"
    )
    assert document["bundle"]["path"] == relative
    assert str(bundle) not in json.dumps(document)


def test_mock_host_runs_the_decision(tmp_path, mock_host):
    bundle, document = _stiff_decision(tmp_path)
    host_json = tmp_path / "host.json"
    completed = _run_host(
        mock_host, bundle, tmp_path / "out.bin", host_json, document["host_args"])
    assert completed.returncode == 0, completed.stderr
    host_doc = json.loads(host_json.read_text())
    board = document["board"]
    chain = board["chain_quanta"]
    assert host_doc["k0_start_quanta"] == document["bundle"]["k0_quanta"]
    assert host_doc["k0_floor_quanta"] == board["floor_quanta"]
    assert host_doc["decay_segment"] == board["segment"]
    assert host_doc["decay_levels"] == len(chain)
    assert host_doc["k0_final_effective"] == board["final_effective"]
    assert host_doc["decay_chain_quanta"] == chain


def test_mock_host_pinned_schedules(tmp_path, mock_host):
    contract = load_contract()
    cases = (
        (50641056, "0.375", "10", 6291456, 4, "0.375",
         [50641056, 25320528, 12660264, 6291456]),
        (317556, "0.002", "400", 33554, 4, "0.001999974250793457",
         [317556, 158778, 79389, 33554]),
    )
    for start_q, floor, segment, floor_q, levels, final, chain in cases:
        bundle = tmp_path / f"pinned_{start_q}.bin"
        _write_with_k0(bundle, contract.dequantize(start_q), iterations=2000)
        host_json = tmp_path / f"pinned_{start_q}.json"
        completed = _run_host(
            mock_host, bundle, tmp_path / f"pinned_{start_q}.out", host_json,
            ["--k0-floor", floor, "--decay-segment", segment])
        assert completed.returncode == 0, completed.stderr
        text = host_json.read_text()
        host_doc = json.loads(text)
        assert host_doc["k0_start_quanta"] == start_q
        assert host_doc["k0_floor_quanta"] == floor_q
        assert host_doc["decay_segment"] == int(segment)
        assert host_doc["decay_levels"] == levels
        assert f'"k0_final_effective":{final}' in text
        assert host_doc["decay_chain_quanta"] == chain


def test_floor_zero_keeps_constant_schedule(tmp_path, mock_host):
    bundle = tmp_path / "zero.bin"
    _write_with_k0(bundle, 0.05, iterations=32)
    plain = tmp_path / "plain.json"
    explicit = tmp_path / "explicit.json"
    first = _run_host(mock_host, bundle, tmp_path / "plain.out", plain, [])
    second = _run_host(
        mock_host, bundle, tmp_path / "explicit.out", explicit, ["--k0-floor", "0"])
    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    plain_doc = json.loads(plain.read_text())
    explicit_doc = json.loads(explicit.read_text())
    assert explicit_doc["k0_floor"] == 0
    assert explicit_doc["decay_segment"] == 0
    assert explicit_doc["decay_levels"] == 1
    assert explicit_doc["decay_chain_quanta"] == []
    assert explicit_doc["decay_chain"] == []
    assert explicit_doc["k0_floor_quanta"] == 0
    timing = {
        "payload_write_ns", "sync_to_device_ns", "mailbox_publish_ns",
        "mailbox_wait_ns", "doorbell_to_done_ns", "kernel_ns",
        "output_sync_read_ns", "complete_ns", "total_ns", "sequence",
        "clock_resolution_ns",
    }
    added = {
        "k0_start_effective", "k0_start_quanta", "k0_floor_effective",
        "k0_floor_quanta", "decay_segment_defaulted", "decay_chain_quanta",
        "decay_chain",
    }
    for key, value in plain_doc.items():
        if key in timing or key in added:
            continue
        assert explicit_doc[key] == value


def test_host_rejects_near_floor_band(tmp_path, mock_host):
    contract = load_contract()
    bundle = tmp_path / "band.bin"
    _write_with_k0(bundle, contract.dequantize(192), iterations=32)
    completed = _run_host(
        mock_host, bundle, tmp_path / "band.out", tmp_path / "band.json",
        ["--k0-floor", repr(contract.dequantize(128)), "--decay-segment", "4"])
    assert completed.returncode == 2
    assert "near-floor band" in completed.stderr


def test_native_chain_parity(tmp_path, native_bin):
    bundle, document = _stiff_decision(tmp_path)
    completed = subprocess.run(
        [str(native_bin), str(bundle), str(tmp_path / "native.out"),
         *document["host_args"]],
        check=False, capture_output=True, text=True)
    assert "DECAY_CHAIN" in completed.stdout
    assert "match=true" in completed.stdout
    assert completed.returncode == 0, completed.stderr


def test_native_odd_level_truncates(tmp_path, native_bin):
    contract = load_contract()
    bundle = tmp_path / "odd.bin"
    _write_with_k0(bundle, contract.dequantize(317556), iterations=2000)
    floor = contract.dequantize(6000)
    completed = subprocess.run(
        [str(native_bin), str(bundle), str(tmp_path / "odd.out"),
         "--k0-floor", repr(floor), "--decay-segment", "285"],
        check=False, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert "DECAY_CHAIN" in completed.stdout
    assert "match=true" in completed.stdout
    assert "19847,9923,6000]" in completed.stdout
    assert "9924" not in completed.stdout


def test_decision_compare_against_native_record(tmp_path, mock_host, native_bin):
    bundle, document = _stiff_decision(tmp_path)
    decision = tmp_path / "decision.json"
    decision.write_text(json.dumps(document))
    native_out = tmp_path / "native.out"
    native = subprocess.run(
        [str(native_bin), str(bundle), str(native_out), *document["host_args"]],
        check=False, capture_output=True, text=True)
    assert native.returncode == 0, native.stderr
    host_json = tmp_path / "host.json"
    host_out = tmp_path / "host.out"
    host = _run_host(mock_host, bundle, host_out, host_json, document["host_args"])
    assert host.returncode == 0, host.stderr
    compared = subprocess.run(
        [sys.executable, str(PKG / "tools" / "board" / "cmp_out.py"),
         str(native_out), str(host_out),
         "--json", str(host_json), "--decision", str(decision)],
        check=False, capture_output=True, text=True)
    # The mock record is not the kernel state, so the byte compare differs.
    # The decision fields are the schedule the compare must accept.
    for field in (
        "k0_start_quanta", "k0_floor_quanta", "decay_segment",
        "decay_levels", "k0_final_effective", "decay_chain_quanta",
    ):
        assert f"{field}:" not in compared.stdout
    assert compared.returncode == 1
    assert "raw" in compared.stdout
    # The same pair without dumps must MATCH, so the option cannot be ignored.
    schedule_only = subprocess.run(
        [sys.executable, str(PKG / "tools" / "board" / "cmp_out.py"),
         "--json", str(host_json), "--decision", str(decision)],
        check=False, capture_output=True, text=True)
    assert schedule_only.returncode == 0, schedule_only.stdout
    assert "MATCH" in schedule_only.stdout


def test_decision_compare_rejects_missing_and_null(tmp_path):
    cmp = PKG / "tools" / "board" / "cmp_out.py"
    final = 5.0 / float(1 << 24)
    decision = {
        "bundle": {"k0_quanta": 10},
        "board": {"chain_quanta": [10, 5], "floor_quanta": 5,
                  "segment": 3, "final_effective": final},
    }
    host = {
        "k0_start_quanta": 10, "k0_floor_quanta": 5, "decay_segment": 3,
        "decay_levels": 2, "k0_final_effective": final,
        "decay_chain_quanta": [10, 5],
    }

    def run(host_doc, decision_doc):
        hp = tmp_path / "host.json"
        dp = tmp_path / "decision.json"
        hp.write_text(json.dumps(host_doc))
        dp.write_text(json.dumps(decision_doc))
        return subprocess.run(
            [sys.executable, str(cmp), "--json", str(hp),
             "--decision", str(dp)],
            check=False, capture_output=True, text=True)

    good = run(host, decision)
    assert good.returncode == 0, good.stdout
    assert "MATCH" in good.stdout

    corrupt = run(
        host, {**decision, "board": {**decision["board"], "floor_quanta": 6}})
    assert corrupt.returncode == 1
    assert "k0_floor_quanta" in corrupt.stdout

    empty = run({}, {})
    assert empty.returncode == 1
    assert "no board object" in empty.stdout

    missing = run(
        {k: v for k, v in host.items() if k != "k0_floor_quanta"}, decision)
    assert missing.returncode == 1
    assert "k0_floor_quanta missing in host" in missing.stdout

    nulled = run({**host, "k0_floor_quanta": None}, decision)
    assert nulled.returncode == 1
    assert "k0_floor_quanta missing in host" in nulled.stdout


def test_cli_input_errors_exit_two(tmp_path):
    from fpga.kv260_v09_bram.tools.board.decay_choice import main
    truncated = tmp_path / "truncated.bin"
    truncated.write_bytes(b"MSRPDL1\0")
    assert main(["--bundle", str(truncated), "--quiet"]) == 2
    easy = tmp_path / "easy.bin"
    _write_with_k0(easy, 0.05, iterations=32)
    assert main([
        "--bundle", str(easy),
        "--json-out", str(tmp_path / "missing" / "decision.json"),
        "--quiet",
    ]) == 2


def test_cli_contract_path_takes_a_path_not_an_object(tmp_path, monkeypatch):
    from fpga.kv260_v09_bram.tools.board.decay_choice import main

    bundle = tmp_path / "contract-path.bin"
    _write_with_k0(bundle, 0.05, iterations=32)
    _patch_recommend(
        monkeypatch, _recommendation(None, "target_already_met", meets=False))
    assert main([
        "--bundle", str(bundle),
        "--contract-path", str(PKG / "env.yaml"),
        "--quiet",
    ]) == 3
    assert main([
        "--bundle", str(bundle),
        "--contract-path", str(tmp_path / "missing.yaml"),
        "--quiet",
    ]) == 2


def test_full_compare_unreadable_json_exits_two(tmp_path):
    cmp = PKG / "tools" / "board" / "cmp_out.py"
    # A board dump with n=0 needs the magic, two uint32s and 16 telemetry words.
    record = b"MSRPFX1\0" + struct.pack("<2I", 1, 0) + b"\0" * (8 * 16)
    ref = tmp_path / "ref.out"
    cand = tmp_path / "cand.out"
    ref.write_bytes(record)
    cand.write_bytes(record)
    malformed = tmp_path / "host.json"
    malformed.write_text("{not json")
    good_json = tmp_path / "good.json"
    good_json.write_text("{}")
    malformed_decision = tmp_path / "decision.json"
    malformed_decision.write_text("[not json")

    def run(*extra):
        return subprocess.run(
            [sys.executable, str(cmp), str(ref), str(cand), *extra],
            check=False, capture_output=True, text=True)

    for json_arg in (str(malformed), str(tmp_path / "missing.json")):
        completed = run("--json", json_arg)
        assert completed.returncode == 2, completed.stderr
        assert "cmp_out.py:" in completed.stderr
        assert "Traceback" not in completed.stderr

    completed = run(
        "--json", str(good_json), "--decision", str(malformed_decision))
    assert completed.returncode == 2, completed.stderr
    assert "cmp_out.py:" in completed.stderr
    assert "Traceback" not in completed.stderr

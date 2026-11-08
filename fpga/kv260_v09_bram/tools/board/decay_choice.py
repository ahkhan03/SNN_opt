"""Choose a quantized k0 floor for one problem bundle.

The integer chain is the schedule the host runs. A confirmation solve at the
exact board floor and segment gates the decision. A level-count difference
against the float staircase is recorded and does not replace that chain.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import numbers
import subprocess
import sys
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np

import snn_opt
from snn_opt import OptimizationProblem, SNNSolver, SolverConfig, recommend_k0_decay

from .bundle import read_bundle
from .decay_contract import final_level, in_near_floor_band, load_contract

SCHEMA = "snn-qp-decay-decision-v1"
_NO_DECAY_REASONS = frozenset({
    "target_already_met",
    "residual_improves_with_iterations",
    "residual_not_linear_in_k0",
})


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        number = float(value)
        if not math.isfinite(number):
            return None
        return number
    if value is None or isinstance(value, str):
        return value
    return str(value)


def _package_version():
    version = getattr(snn_opt, "__version__", None)
    return None if version is None else str(version)


def _package_commit():
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    text = completed.stdout.strip()
    return text or None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _empty_board():
    return {
        "floor_requested": None,
        "floor_quanta": None,
        "floor_effective": None,
        "quantization_error": None,
        "clamped_to_min": False,
        "levels_capped": False,
        "chain_quanta": [],
        "chain": [],
        "segment": None,
        "segment_adjusted": False,
        "final_effective": None,
        "float_levels": None,
        "board_levels": 0,
        "level_count_matches": None,
    }


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _bundle_identity(path: Path) -> dict:
    """Basename and digest. Never an absolute filesystem path.

    ``path`` is the basename. When the bundle lies inside this repository
    it is the repository-relative path instead. ``sha256`` is always set.
    """
    resolved = Path(path).resolve()
    name = resolved.name
    try:
        recorded = resolved.relative_to(_repo_root()).as_posix()
    except ValueError:
        recorded = name
    return {"name": name, "path": recorded, "sha256": _sha256(resolved)}


def _document(path, bundle, contract, start_q, start_mode, recommendation):
    evidence = recommendation.evidence
    return {
        "schema": SCHEMA,
        "package": {"version": _package_version(), "commit": _package_commit()},
        "bundle": {
            **_bundle_identity(path),
            "n": int(bundle["n"]),
            "m": int(bundle["m"]),
            "iterations": int(bundle["iters"]),
            "k0": float(bundle["k0"]),
            "k0_quanta": int(start_q),
            "projection_cap": int(bundle["cap"]),
            "constraint_tol": float(bundle["tol"]),
            "has_lower": bool(bundle["has_lower"]),
            "has_upper": bool(bundle["has_upper"]),
            "start_mode": start_mode,
        },
        "contract": {
            "word_bits": contract.word_bits,
            "int_bits": contract.int_bits,
            "fraction_bits": contract.fraction_bits,
            "min_floor_quanta": contract.min_floor_quanta,
            "min_floor": contract.min_floor,
        },
        "recommendation": {
            "floor": None if recommendation.floor is None else float(recommendation.floor),
            "segment": None if recommendation.segment is None else int(recommendation.segment),
            "levels": [float(level) for level in recommendation.levels],
            "reason": recommendation.reason,
            "meets_target": bool(evidence.get("meets_target")),
            "evidence": _jsonable(evidence),
        },
        "board": _empty_board(),
        "confirmation": {
            "kkt_normalized": None,
            "kkt_fit_status": None,
            "kkt_residual": None,
            "kkt_scale": None,
            "convergence_reason": None,
            "converged": None,
            "projection_budget_exhausted": None,
            "certificate_available": False,
            "meets_target": None,
        },
        "decision": "refuse",
        "host_args": [],
    }


def _finite_number(value):
    return isinstance(value, numbers.Real) and math.isfinite(float(value))


def _confirm(document, problem, config, x_start, start_eff, floor_eff, segment, target):
    board = document["board"]
    try:
        # The kernel runs every iteration. Measurement solves do the same,
        # so the certificate is not an early-stopped prefix of that budget.
        applied = replace(
            config,
            k0=float(start_eff),
            k0_decay_floor=float(floor_eff),
            k0_decay_segment=int(segment),
            convergence=replace(config.convergence, enable_early_stopping=False),
        )
        result = SNNSolver(problem, applied).solve(x_start, verbose=False)
    except ValueError as exc:
        # The float staircase can reject a schedule the integer chain accepts.
        # The integer chain stays the schedule; the missing certificate refuses.
        board["float_levels"] = None
        board["level_count_matches"] = False
        document["confirmation"] = {
            "kkt_normalized": None,
            "kkt_fit_status": "not_computed",
            "kkt_residual": None,
            "kkt_scale": None,
            "convergence_reason": "solver_rejected",
            "converged": None,
            "projection_budget_exhausted": None,
            "certificate_available": False,
            "meets_target": False,
            "detail": str(exc),
        }
        return
    kkt = getattr(result, "kkt_normalized", None)
    kkt_value = float(kkt) if _finite_number(kkt) else None
    residual = getattr(result, "kkt_residual", None)
    scale = getattr(result, "kkt_scale", None)
    fit_status = str(getattr(result, "kkt_fit_status", "") or "")
    # A zero residual at zero scale is an exact pass: the solver's gate has no
    # additive term, and the normalized ratio is 0/0 there.
    zero_scale_pass = (
        fit_status == "ok"
        and _finite_number(residual) and float(residual) == 0.0
        and _finite_number(scale) and float(scale) == 0.0)
    aborted = bool(getattr(result, "projection_budget_exhausted", False))
    certificate = bool(
        fit_status == "ok" and (kkt_value is not None or zero_scale_pass))
    # Fail closed: an aborted sweep or an unavailable certificate is never a
    # pass, however small the number that accompanies it.
    meets = bool(
        certificate and not aborted
        and (zero_scale_pass
             or (kkt_value is not None and kkt_value <= float(target))))
    executed = getattr(result, "decay_levels", None)
    float_levels = None if executed is None else len(executed)
    board["float_levels"] = float_levels
    board["level_count_matches"] = float_levels == board["board_levels"]
    document["confirmation"] = {
        "kkt_normalized": kkt_value,
        "kkt_fit_status": fit_status or None,
        "kkt_residual": float(residual) if _finite_number(residual) else None,
        "kkt_scale": float(scale) if _finite_number(scale) else None,
        "convergence_reason": (
            str(getattr(result, "convergence_reason", "") or "") or None),
        "converged": bool(getattr(result, "converged", False)),
        "projection_budget_exhausted": aborted,
        "certificate_available": certificate,
        "meets_target": meets,
    }


def decide_for_bundle(
    bundle_path,
    *,
    target=1e-4,
    floor_margin=0.5,
    max_verifications=3,
    probe_iterations=None,
    backend="python",
    allow_above_target=False,
    start="bundle",
    contract=None,
):
    """Return ``(document, exit_code)`` for one bundle.

    Exit 0 writes a decay the host can run. Exit 3 is ``no_decay`` or
    ``refuse``; the JSON ``decision`` field distinguishes them. Callers turn
    every other failure into exit 2.

    ``contract`` is a loaded contract object (``load_contract()``) or
    ``None`` for the shipped ``env.yaml``. It is never a path: the CLI
    resolves ``--contract-path`` before calling, so the argument has one
    type.
    """
    path = Path(bundle_path)
    bundle = read_bundle(path)
    if contract is None:
        contract = load_contract()
    problem = OptimizationProblem(
        A=bundle["A"], b=bundle["b"], C=bundle["C"], d=bundle["d"])
    config = SolverConfig(
        k0=float(bundle["k0"]),
        max_iterations=int(bundle["iters"]),
        max_projection_iters=int(bundle["cap"]),
        constraint_tol=float(bundle["tol"]),
        lower_bound=bundle["lower"] if bundle["has_lower"] else None,
        upper_bound=bundle["upper"] if bundle["has_upper"] else None,
        integration_method="euler",
        transform=None,
        backend=backend,
    )
    x_start = None if start == "zero" else bundle["x0"]
    recommendation = recommend_k0_decay(
        problem,
        config,
        target=float(target),
        x0=x_start,
        probe_iterations=probe_iterations,
        floor_margin=float(floor_margin),
        min_floor=contract.min_floor,
        max_verifications=int(max_verifications),
        max_levels=32,
    )
    start_q = contract.quantize(float(config.k0))
    document = _document(path, bundle, contract, start_q, start, recommendation)
    board = document["board"]
    evidence = recommendation.evidence
    board["clamped_to_min"] = bool(evidence.get("floor_clamped_to_min"))
    board["levels_capped"] = bool(evidence.get("levels_capped"))
    if board["clamped_to_min"]:
        warnings.warn(
            "k0 floor clamped to the fixed-point minimum; "
            "confirmation at that floor gates the decision",
            stacklevel=2,
        )
    if recommendation.floor is None:
        document["decision"] = (
            "no_decay" if recommendation.reason in _NO_DECAY_REASONS else "refuse")
        document["host_args"] = []
        return document, 3

    floor_q = contract.quantize(float(recommendation.floor))
    floor_eff = contract.dequantize(floor_q)
    board["floor_requested"] = float(recommendation.floor)
    board["floor_quanta"] = floor_q
    board["floor_effective"] = floor_eff
    # Signed distance from the requested floor to the quantum it landed on.
    board["quantization_error"] = float(recommendation.floor) - floor_eff
    if not (contract.min_floor_quanta <= floor_q < start_q):
        document["decision"] = "refuse"
        document["recommendation"]["reason"] = (
            f"quantized floor {floor_q} is outside "
            f"[{contract.min_floor_quanta}, {start_q})"
        )
        document["host_args"] = []
        return document, 3
    if in_near_floor_band(start_q, floor_q):
        document["decision"] = "refuse"
        document["recommendation"]["reason"] = "floor_too_close_to_start"
        document["host_args"] = []
        return document, 3

    chain_q = contract.chain(start_q, floor_q)
    iterations = int(bundle["iters"])
    segment = iterations // len(chain_q)
    board["chain_quanta"] = [int(quanta) for quanta in chain_q]
    board["chain"] = [contract.dequantize(quanta) for quanta in chain_q]
    board["segment"] = int(segment)
    board["segment_adjusted"] = (
        recommendation.segment is not None and int(recommendation.segment) != int(segment)
    )
    board["board_levels"] = len(chain_q)
    if segment < 1 or segment * len(chain_q) > iterations:
        document["decision"] = "refuse"
        document["recommendation"]["reason"] = (
            f"segment {segment} does not cover {len(chain_q)} levels "
            f"within {iterations} iterations"
        )
        document["host_args"] = []
        return document, 3

    start_eff = contract.dequantize(start_q)
    board["final_effective"] = contract.dequantize(
        final_level(start_q, floor_q, segment, iterations))
    solve_x = bundle["x0"] if start == "bundle" else _zeros(bundle["n"])
    _confirm(document, problem, config, solve_x, start_eff, floor_eff, segment, target)

    # recommendation.meets_target stays the probe result. The gate is only
    # confirmation.meets_target. None and false are above-target.
    confirmed = document["confirmation"]["meets_target"]
    if confirmed is True or allow_above_target:
        document["decision"] = "decay"
        document["host_args"] = [
            "--k0-floor",
            repr(floor_eff),
            "--decay-segment",
            str(int(segment)),
        ]
        return document, 0
    document["decision"] = "refuse"
    document["host_args"] = []
    return document, 3


def _zeros(n):
    return np.zeros(int(n), dtype=float)


def _write_outputs(document, json_out, args_out):
    if json_out:
        Path(json_out).write_text(json.dumps(document, indent=2) + "\n")
    if args_out:
        lines = document["host_args"]
        text = ("\n".join(lines) + "\n") if lines else ""
        Path(args_out).write_text(text)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="decay_choice",
        description="Quantize a recommended k0 floor for the v09b host.",
    )
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--target", type=float, default=1e-4)
    parser.add_argument("--floor-margin", type=float, default=0.5)
    parser.add_argument("--max-verifications", type=int, default=3)
    parser.add_argument("--probe-iterations", type=int, default=None)
    parser.add_argument("--backend", choices=("python", "c"), default="python")
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--args-out", default=None)
    parser.add_argument("--allow-above-target", action="store_true")
    parser.add_argument("--start", choices=("bundle", "zero"), default="bundle")
    parser.add_argument(
        "--contract-path",
        default=None,
        help="contract YAML to use instead of the shipped env.yaml",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    try:
        contract = (
            load_contract(args.contract_path)
            if args.contract_path is not None
            else None
        )
        document, code = decide_for_bundle(
            args.bundle,
            target=args.target,
            floor_margin=args.floor_margin,
            max_verifications=args.max_verifications,
            probe_iterations=args.probe_iterations,
            backend=args.backend,
            allow_above_target=args.allow_above_target,
            start=args.start,
            contract=contract,
        )
        _write_outputs(document, args.json_out, args.args_out)
    except Exception as exc:  # noqa: BLE001 - the CLI contract is exit 2
        print(f"decay_choice: {exc}", file=sys.stderr)
        return 2
    if not args.quiet:
        reason = document["recommendation"]["reason"]
        print(f"decision={document['decision']} reason={reason}")
    return code


if __name__ == "__main__":
    sys.exit(main())

"""Explicit k0-decay recommendation.

The solver contract is explicit: ``SolverConfig.k0_decay_floor`` is an
absolute step size or ``None`` (constant k0). This module *proposes* a floor.
It never changes a default, never runs the schedule inside ``solve``, and
never edits the config it is handed. The caller passes the returned float
back through ``k0_decay_floor`` and records the recommendation (and its
evidence) in the script or the paper.

Why a recommender rather than an ``auto`` default
------------------------------------------------
The offset at a spurious fixed point is ``O(k0)``: halving ``k0`` halves the
normalized KKT residual at an *unchanged* iteration budget. That is a bias,
not slow convergence, so more iterations do not remove it and a smaller step
does. But the signature is a property of a particular problem, and the
optimal floor depends on the accuracy target, so no single default is right
for every caller. Mature optimizer APIs therefore either make a step
schedule an explicit object or expose an adaptive mode only as a named,
documented, overridable choice whose decision is reported. This helper
follows the second pattern without touching ``solve``: it decides, it reports
why, and the caller opts in.

How the decision is made
------------------------
1. **Signature probe.** Solve with constant ``k0`` at the configured start
   on a bounded budget. If the normalized KKT residual is already at or
   below ``target``, decay is unnecessary. Otherwise the residual is run
   again at the same ``k0`` with twice the budget (a plateau that more
   iterations do not move) and at half ``k0`` (a residual that halving the
   step cuts by at least 30 percent). Both are the ``O(k0)`` signature.
2. **Model floor.** With a linear bias the residual at ``k`` is
   ``residual_at_start * k / k0_start``, so reaching ``target`` needs
   ``k <= k0_start * target / residual_at_start``. ``floor_margin`` shrinks
   that estimate for model error.
3. **Bounded verification.** The model is an extrapolation, so the actual
   staircase is run at the candidate floor with the caller's iteration
   budget and its measured residual decides. If the target is still missed
   the floor is halved and retried, up to ``max_verifications`` solves.
   Every attempt is reported, so the caller can see what was tried and what
   it cost. A run that aborts on the projection watchdog, or that stops
   before reaching the candidate floor, is never accepted however small its
   residual happens to be at the abort point.

The probe mirrors a real decay run's machinery: it raises the inner-sweep
watchdog through the same helper the solver uses (``decay=True``), and it
starts from the caller's ``x0`` when one is supplied. Measurement solves run
with early stopping disabled, so each consumes its specified budget and the
comparison between two steps is a comparison of their fixed points rather
than of two unrelated early exits; the caller's convergence settings are not
modified. A cold start is out of scope for decay (it can exhaust the
projection sweep before the first level even at the raised cap), so a probe
that exhausts the sweep returns no recommendation instead of guessing.

Example
-------
>>> from dataclasses import replace
>>> rec = recommend_k0_decay(problem, config, target=1e-4, x0=warm_start)
>>> if rec.floor is not None:
...     config = replace(config, k0_decay_floor=rec.floor,
...                      k0_decay_segment=rec.segment)
>>> result = SNNSolver(problem, config).solve(warm_start)

Nothing here runs on the constant-k0 path: with ``k0_decay_floor=None`` the
solve is byte-identical to the pre-decay solver, whether or not a
recommendation was computed.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional

import numpy as np

from .solver import (
    OptimizationProblem,
    SNNSolver,
    SolverConfig,
    _automatic_projection_cap,
    _effective_kkt_target,
    _k0_decay_levels,
)

#: Probe budget when the caller does not set one. The fixed point of the
#: Euler/projection map is reached well inside this on the measured stiff
#: problems (500 iterations already matches 50,000 there), so a larger
#: default would only multiply the probe cost.
DEFAULT_PROBE_ITERATIONS = 1000

#: A residual that more iterations improve by more than this fraction is
#: converging, not biased; decay is not the indicated fix.
_PLATEAU_IMPROVEMENT = 0.9

#: Halving ``k0`` must cut the normalized residual to at most this fraction.
#: True ``O(k0)`` behaviour is about one half; the band is loose enough for
#: measurement noise and strict enough to reject a residual that ignores k0.
_LINEARITY_MAX_RATIO = 0.7


@dataclass(frozen=True)
class DecayRecommendation:
    """A proposed staircase and the evidence behind it.

    Attributes
    ----------
    floor : float or None
        Absolute ``k0`` floor, in the same units as ``SolverConfig.k0``.
        ``None`` means the probe found no ``O(k0)`` bias to remove (or the
        probe could not complete); ``evidence['reason']`` names which.
    segment : int or None
        Outer iterations per level, the solver's automatic split
        (``max_iterations // n_levels``). None when ``floor`` is None.
    levels : list of float
        The staircase this floor generates, starting at the configured
        effective ``k0`` and ending exactly at ``floor``. Empty when
        ``floor`` is None.
    evidence : dict
        Probe and verification measurements: the residuals at each probed
        step, the plateau and linearity ratios, every candidate floor that
        was run, the achieved residual, the target, and the cost (number of
        solves, iterations, wall seconds). ``evidence['reason']`` is the
        decision label.
    reason : str
        The decision label, duplicated from ``evidence['reason']`` for
        convenience.
    """

    floor: Optional[float]
    segment: Optional[int]
    levels: List[float]
    evidence: Dict[str, Any]
    reason: str


def _probe_record(k0: float, iterations: int, result) -> Dict[str, Any]:
    """One measurement solve, reduced to the fields the decision uses."""
    return {
        "k0": float(k0),
        "iterations": int(iterations),
        "iterations_used": int(result.iterations_used),
        "kkt_residual": float(result.kkt_residual),
        "kkt_scale": float(result.kkt_scale),
        "kkt_normalized": float(result.kkt_normalized),
        "kkt_fit_status": str(result.kkt_fit_status),
        "converged": bool(result.converged),
        "convergence_reason": str(result.convergence_reason),
        "projection_budget_exhausted": bool(result.projection_budget_exhausted),
    }


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _zero_scale_pass(record: Dict[str, Any]) -> bool:
    """A zero-residual, zero-scale certificate is an exact pass.

    The solver's gate is ``residual <= target * scale`` with no additive
    term, so at scale 0 it accepts only a residual of exactly 0. The
    normalized ratio is then 0/0, which is a pass, not missing evidence.
    """
    return (record.get("kkt_fit_status") == "ok"
            and _finite(record.get("kkt_residual"))
            and float(record["kkt_residual"]) == 0.0
            and _finite(record.get("kkt_scale"))
            and float(record["kkt_scale"]) == 0.0)


def _certificate_available(record: Dict[str, Any]) -> bool:
    """Whether the certificate can be read at all.

    A non-``ok`` fit status makes the certificate unavailable whatever the
    accompanying numbers are: the solver's own gate requires ``ok`` and fails
    closed, and the recommender must not accept a diagnostic the solver would
    reject.
    """
    if record.get("kkt_fit_status") != "ok":
        return False
    return _finite(record.get("kkt_normalized")) or _zero_scale_pass(record)


def _certificate_met(record: Dict[str, Any], target: float) -> bool:
    """Whether a measurement solve's KKT evidence meets ``target``."""
    if not _certificate_available(record):
        return False
    if _finite(record.get("kkt_normalized")):
        return float(record["kkt_normalized"]) <= float(target)
    return _zero_scale_pass(record)


def recommend_k0_decay(
    problem: OptimizationProblem,
    config: Optional[SolverConfig] = None,
    target: Optional[float] = None,
    *,
    x0: Optional[np.ndarray] = None,
    probe_iterations: Optional[int] = None,
    floor_margin: float = 0.5,
    min_floor: Optional[float] = None,
    max_levels: int = 32,
    max_verifications: int = 3,
    check_plateau: bool = True,
) -> DecayRecommendation:
    """Propose an absolute ``k0_decay_floor`` from a constant-``k0`` probe.

    Parameters
    ----------
    problem : OptimizationProblem
        The problem the staircase would run on.
    config : SolverConfig, optional
        The configuration the caller intends to use. Its ``k0``/``k0_scale``
        fix the start of the staircase, its ``max_iterations`` fixes the
        budget, and its ``convergence.kkt_target`` is the default target.
        The object is not mutated, and its convergence settings do not gate
        the measurement solves.
    target : float, optional
        Normalized KKT target the floor should reach. Defaults to the
        config's effective KKT target (``1e-4`` when unset).
    x0 : ndarray, optional
        Starting point. Defaults to the zero vector. Pass the warm start the
        real solve will use: the probe measures the fixed point from there.
    probe_iterations : int, optional
        Budget for the signature probes. Defaults to
        ``min(max_iterations, 1000)``. The plateau probe runs at twice this
        budget.
    floor_margin : float
        Multiplier on the model floor, in ``(0, 1]``. Below 1 the floor is
        made smaller (more conservative); each halving costs exactly one
        extra level.
    min_floor : float, optional
        Absolute lower bound for the returned floor, for a fixed-point
        backend whose step resolution has a hard lower bound. A clamped
        floor is flagged in the evidence.
    max_levels : int
        Cap on the number of staircase levels. A model floor that would
        need more is raised to fit, and the clamp is reported.
    max_verifications : int
        Total number of staircase solves allowed to confirm the target
        (the first candidate plus the halvings after each miss). At least 1.
    check_plateau : bool
        Run the doubled-budget plateau probe. Disable to save one solve when
        the caller already knows the residual does not move with iterations;
        the linearity probe then carries the decision alone.

    Returns
    -------
    DecayRecommendation
        ``floor is None`` when decay is not indicated, with the reason and
        the measurements in ``evidence``. A returned floor may be flagged
        ``meets_target=False`` in the evidence (reason
        ``recommended_above_target``) when the caller's budget, a level cap,
        ``min_floor``, or model error prevents reaching the target.

    Notes
    -----
    The recommender presumes a warm start and a ``k0_decay_floor`` that the
    solver can accept (``0 < floor < k0_eff``). It judges bias with the
    normalized KKT residual, which is the solver's own acceptance gate, and
    it never returns a floor below ``min_floor``. It does not run the
    schedule for the caller; apply ``floor`` and ``segment`` explicitly.
    """
    config = SolverConfig() if config is None else config
    if config.integration_method != "euler":
        raise ValueError(
            "recommend_k0_decay requires integration_method='euler'; got "
            f"{config.integration_method!r}. k0_decay_floor is an Euler "
            "schedule and is rejected by the solver on other methods.")
    if config.transform is not None:
        raise ValueError(
            "recommend_k0_decay cannot probe a transformed problem: the "
            "transform applies k0 in transformed coordinates, so a floor "
            "measured on the original problem would not transfer. Recommend "
            "on the untransformed problem, or set the floor by hand."
        )

    target_value = (float(target) if target is not None
                    else _effective_kkt_target(config.convergence.kkt_target))
    if not (_finite(target_value) and target_value > 0.0):
        raise ValueError(
            f"target must be finite and > 0, got {target_value!r}")
    if not (_finite(floor_margin) and 0.0 < float(floor_margin) <= 1.0):
        raise ValueError(
            "floor_margin must be in (0, 1], got "
            f"{floor_margin!r}")
    if min_floor is not None and not (_finite(min_floor) and min_floor > 0.0):
        raise ValueError(
            f"min_floor must be finite and > 0 when set, got {min_floor!r}")
    if int(max_levels) < 2:
        raise ValueError(f"max_levels must be >= 2, got {max_levels!r}")
    if int(max_verifications) < 1:
        raise ValueError(
            "max_verifications must be >= 1: the recommendation is measured, "
            f"so at least one verification solve is required; got "
            f"{max_verifications!r}")

    max_iterations = int(config.max_iterations)
    if max_iterations < 1:
        raise ValueError(
            f"config.max_iterations must be >= 1, got {max_iterations!r}")

    if probe_iterations is None:
        probe_iterations = min(max_iterations, DEFAULT_PROBE_ITERATIONS)
    probe_iterations = int(probe_iterations)
    if probe_iterations < 1:
        raise ValueError(
            f"probe_iterations must be >= 1, got {probe_iterations!r}")

    x_start = (np.zeros(problem.n_vars, dtype=float)
               if x0 is None else np.asarray(x0, dtype=float).copy())
    if x_start.shape != (problem.n_vars,):
        raise ValueError(
            f"x0 must have shape ({problem.n_vars},), got {x_start.shape}")

    # The effective start of the staircase. One solve-time computation, not a
    # second implementation of k0_scale / L.
    k0_eff = float(SNNSolver(problem, config)._k0)

    n_candidates = len(getattr(problem, "nonlinear_candidates", ()) or ())
    probe_cap = config.max_projection_iters
    if probe_cap is None:
        probe_cap = _automatic_projection_cap(
            problem, config, n_candidates=n_candidates, decay=True)

    # Measurement solves consume their full budget: an early exit at the
    # caller's gate would compare two arbitrary stopping points instead of
    # two fixed points, and a target tighter than that gate would never be
    # measured at all. The caller's object is copied, never edited.
    measure_convergence = replace(
        config.convergence, enable_early_stopping=False)

    evidence: Dict[str, Any] = {
        "target": float(target_value),
        "k0_start": k0_eff,
        "max_iterations": max_iterations,
        "probe_iterations": probe_iterations,
        "probe_start": "zero" if x0 is None else "caller",
        "projection_cap": int(probe_cap),
        "probe_solves": 0,
        "probe_iterations_total": 0,
        "verification_solves": 0,
        "verification_iterations_total": 0,
        "wall_seconds": 0.0,
        "probes": [],
        "verifications": [],
        "floor_margin": float(floor_margin),
        "min_floor": None if min_floor is None else float(min_floor),
        "max_levels": int(max_levels),
        "floor_clamped_to_min": False,
        "levels_capped": False,
        "plateau_ratio": None,
        "k0_linearity_ratio": None,
        "model_floor": None,
        "predicted_kkt_at_floor": None,
        "achieved_kkt_normalized": None,
        "meets_target": False,
        "n_levels": None,
        "segment": None,
        "reason": "",
    }

    started = time.perf_counter()
    probe_cfg = replace(
        config, k0_decay_floor=None, k0_decay_segment=None,
        max_projection_iters=int(probe_cap),
        convergence=measure_convergence)

    def probe(k0: float, iterations: int) -> Dict[str, Any]:
        cfg = replace(probe_cfg, k0=float(k0), max_iterations=int(iterations))
        result = SNNSolver(problem, cfg).solve(x_start, verbose=False)
        record = _probe_record(k0, iterations, result)
        evidence["probes"].append(record)
        evidence["probe_solves"] += 1
        evidence["probe_iterations_total"] += int(result.iterations_used)
        return record

    def finish(floor: Optional[float], reason: str) -> DecayRecommendation:
        evidence["wall_seconds"] = round(
            time.perf_counter() - started, 6)
        evidence["reason"] = reason
        if floor is None:
            return DecayRecommendation(floor=None, segment=None, levels=[],
                                       evidence=evidence, reason=reason)
        return DecayRecommendation(
            floor=float(floor), segment=int(evidence["segment"]),
            levels=[float(level) for level in evidence["levels"]],
            evidence=evidence, reason=reason)

    # 1. Signature probe at the configured step.
    first = probe(k0_eff, probe_iterations)
    if first["projection_budget_exhausted"]:
        return finish(None, "probe_projection_budget_exhausted")
    if _certificate_met(first, target_value):
        return finish(None, "target_already_met")
    if not _certificate_available(first):
        return finish(None, "kkt_certificate_unavailable")

    # 2. Plateau: the residual is a fixed point, not slow convergence.
    if check_plateau:
        second = probe(k0_eff, 2 * probe_iterations)
        if second["projection_budget_exhausted"]:
            return finish(None, "probe_projection_budget_exhausted")
        if _certificate_met(second, target_value):
            return finish(None, "residual_improves_with_iterations")
        if not _certificate_available(second):
            return finish(None, "kkt_certificate_unavailable")
        plateau_ratio = (second["kkt_normalized"]
                         / first["kkt_normalized"])
        evidence["plateau_ratio"] = float(plateau_ratio)
        if plateau_ratio < _PLATEAU_IMPROVEMENT:
            return finish(None, "residual_improves_with_iterations")

    # 3. Linearity: halving the step cuts the residual.
    halved = probe(k0_eff / 2.0, probe_iterations)
    if halved["projection_budget_exhausted"]:
        return finish(None, "probe_projection_budget_exhausted")
    # The certificate gate comes first: a non-"ok" fit is not evidence of
    # linearity however finite its accompanying diagnostic is.
    if not _certificate_available(halved):
        return finish(None, "kkt_certificate_unavailable")
    if _finite(halved["kkt_normalized"]):
        linear_ratio = halved["kkt_normalized"] / first["kkt_normalized"]
    else:
        # Available with a non-finite ratio means the zero-scale pass: the
        # residual is exactly zero, so the step term vanished.
        linear_ratio = 0.0
    evidence["k0_linearity_ratio"] = float(linear_ratio)
    if linear_ratio > _LINEARITY_MAX_RATIO:
        return finish(None, "residual_not_linear_in_k0")

    # 4. Model floor, then bounded verification of the actual staircase.
    model_floor = k0_eff * target_value / first["kkt_normalized"]
    evidence["model_floor"] = float(model_floor)
    candidate = float(model_floor) * float(floor_margin)
    if min_floor is not None and candidate < float(min_floor):
        candidate = float(min_floor)
        evidence["floor_clamped_to_min"] = True

    attempts: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None
    all_aborted = True
    if not (candidate > 0.0 and candidate < k0_eff):
        return finish(None, "min_floor_at_or_above_start")
    for _ in range(int(max_verifications)):
        if not (candidate > 0.0 and candidate < k0_eff):
            break
        levels = _k0_decay_levels(k0_eff, candidate)
        if len(levels) > int(max_levels):
            candidate = k0_eff / 2.0 ** (int(max_levels) - 1)
            evidence["levels_capped"] = True
            levels = _k0_decay_levels(k0_eff, candidate)
        n_levels = len(levels)
        if n_levels > max_iterations:
            # Same shape as every other attempt, so a consumer that walks
            # ``evidence['verifications']`` never has to special-case it.
            attempts.append({
                "k0": float(candidate),
                "iterations": int(max_iterations),
                "iterations_used": 0,
                "kkt_residual": float("nan"),
                "kkt_scale": float("nan"),
                "kkt_normalized": None,
                "kkt_fit_status": "not_computed",
                "converged": False,
                "convergence_reason": "budget_cannot_reach_floor",
                "projection_budget_exhausted": False,
                "floor": float(candidate),
                "n_levels": int(n_levels),
                "segment": 0,
                "reached_floor": False,
                "decay_levels": None,
                "meets_target": False,
            })
            all_aborted = False
            break
        segment = max_iterations // n_levels
        cfg = replace(config, k0=k0_eff,
                      k0_decay_floor=float(candidate),
                      k0_decay_segment=None,
                      max_iterations=max_iterations,
                      max_projection_iters=int(probe_cap),
                      convergence=measure_convergence)
        result = SNNSolver(problem, cfg).solve(x_start, verbose=False)
        executed = (None if result.decay_levels is None
                    else [float(level) for level in result.decay_levels])
        # The schedule appends the exact floor, so require a relative match
        # only: an absolute tolerance would accept a last level that is many
        # times a very small candidate.
        reached_floor = bool(
            executed
            and math.isclose(executed[-1], float(candidate),
                             rel_tol=1e-9, abs_tol=0.0))
        attempt = _probe_record(candidate, max_iterations, result)
        attempt.update({
            "floor": float(candidate),
            "n_levels": int(n_levels),
            "segment": int(segment),
            "reached_floor": reached_floor,
            "decay_levels": executed,
            "meets_target": False,
        })
        attempts.append(attempt)
        evidence["verification_solves"] += 1
        evidence["verification_iterations_total"] += int(result.iterations_used)
        if not attempt["projection_budget_exhausted"]:
            all_aborted = False
        attempt["meets_target"] = bool(
            not attempt["projection_budget_exhausted"]
            and reached_floor
            and _certificate_met(attempt, target_value))
        if attempt["meets_target"]:
            best = attempt
            break
        usable = (not attempt["projection_budget_exhausted"]
                  and reached_floor and _certificate_available(attempt))
        if usable and (best is None
                       or _finite(attempt["kkt_normalized"])
                       and (not _finite(best["kkt_normalized"])
                            or attempt["kkt_normalized"]
                            < best["kkt_normalized"])):
            best = attempt
        nxt = candidate / 2.0
        if min_floor is not None and nxt < float(min_floor):
            if candidate <= float(min_floor):
                break
            nxt = float(min_floor)
            evidence["floor_clamped_to_min"] = True
        if nxt >= candidate:
            break
        candidate = nxt
        if len(_k0_decay_levels(k0_eff, candidate)) > int(max_levels):
            break

    evidence["verifications"] = attempts
    if all_aborted and attempts:
        return finish(None, "verification_projection_budget_exhausted")
    if best is None:
        return finish(None, "no_usable_floor_within_budget")

    floor = float(best["floor"])
    levels = _k0_decay_levels(k0_eff, floor)
    evidence["n_levels"] = int(len(levels))
    evidence["levels"] = [float(level) for level in levels]
    evidence["segment"] = int(best["segment"])
    evidence["achieved_kkt_normalized"] = best["kkt_normalized"]
    evidence["meets_target"] = bool(best["meets_target"])
    evidence["predicted_kkt_at_floor"] = float(
        first["kkt_normalized"] * floor / k0_eff)
    if not best["meets_target"]:
        return finish(floor, "recommended_above_target")
    return finish(floor, "recommended")

"""Explicit k0-decay recommendation.

The recommender proposes a floor from a constant-k0 probe; it never changes
a default. The two problems below are fixed, reproducible instances: one
whose constant-k0 residual sits at a spurious fixed point above the target
(the O(k0) signature), and one whose residual is already at the target.
"""

from dataclasses import asdict

import numpy as np
import pytest

import snn_opt.recommend as recommend_module
from snn_opt import (
    DecayRecommendation,
    OptimizationProblem,
    SNNSolver,
    SolverConfig,
    recommend_k0_decay,
)

# Stiff instance: constant k0 at the default k0_scale stalls at a normalized
# KKT residual of about 2.2e-2, unchanged between 1000 and 2000 iterations,
# and halving k0 roughly halves it. Three variables, six halfspaces.
_STIFF_A = np.array([
    [0.11387419, 0.32740482, 0.38032890],
    [0.32740482, 3.10892627, 3.23929608],
    [0.38032890, 3.23929608, 3.95096198],
])
_STIFF_b = np.array([0.12279726, -0.81305707, 1.97858988])
_STIFF_C = np.array([
    [-0.44193274, -0.67092762, -0.59544234],
    [0.22153773, -0.88767343, 0.40367922],
    [0.13692966, 0.81618772, -0.56132689],
    [0.99483654, -0.10026541, -0.01571940],
    [-0.85639087, -0.27182262, 0.43898423],
    [-0.26361726, 0.25906587, -0.92918826],
])
_STIFF_d = np.array([
    -0.84749728, -0.43916840, -0.43455156,
    -0.58556949, -0.29355200, -0.32266863,
])

# Healthy instance: the same probe lands at 7.6e-6, below the 1e-4 target.
_HEALTHY_A = np.array([
    [3.70686090, 3.17021637, 1.62443914],
    [3.17021637, 5.30004987, 3.81985579],
    [1.62443914, 3.81985579, 4.36982807],
])
_HEALTHY_b = np.array([-0.03663783, -0.89724926, 0.23313208])
_HEALTHY_C = np.array([
    [-0.67444237, 0.34918978, 0.65053362],
    [-0.24708692, 0.44858512, 0.85890596],
])
_HEALTHY_d = np.array([-0.92114486, -0.37371861])

TARGET = 1e-4


def _problem(A, b, C, d):
    return OptimizationProblem(
        A=np.asarray(A, float), b=np.asarray(b, float),
        C=np.asarray(C, float), d=np.asarray(d, float))


def _stiff():
    return _problem(_STIFF_A, _STIFF_b, _STIFF_C, _STIFF_d)


def _healthy():
    return _problem(_HEALTHY_A, _HEALTHY_b, _HEALTHY_C, _HEALTHY_d)


class TestStiffProblem:
    def test_recommends_a_floor_that_meets_the_target(self):
        config = SolverConfig(max_iterations=8000)
        rec = recommend_k0_decay(_stiff(), config, target=TARGET)
        assert isinstance(rec, DecayRecommendation)
        assert rec.floor is not None
        assert rec.reason == "recommended"
        assert rec.evidence["meets_target"] is True
        assert rec.evidence["achieved_kkt_normalized"] <= TARGET
        # The evidence shows the O(k0) signature that justified the call.
        assert rec.evidence["plateau_ratio"] >= 0.9
        assert rec.evidence["k0_linearity_ratio"] <= 0.7
        assert rec.evidence["k0_start"] > rec.floor > 0.0

    def test_levels_and_segment_match_the_solver_rule(self):
        config = SolverConfig(max_iterations=8000)
        rec = recommend_k0_decay(_stiff(), config, target=TARGET)
        assert rec.levels[0] == pytest.approx(rec.evidence["k0_start"])
        assert rec.levels[-1] == pytest.approx(rec.floor)
        # Every level after the first is a half of its predecessor, except the
        # final floor, which is exact.
        for upper, lower in zip(rec.levels[:-2], rec.levels[1:-1]):
            assert lower == pytest.approx(upper / 2.0)
        assert rec.segment == config.max_iterations // len(rec.levels)

    def test_recommendation_applies_through_the_explicit_contract(self):
        config = SolverConfig(max_iterations=8000)
        rec = recommend_k0_decay(_stiff(), config, target=TARGET)
        from dataclasses import replace
        applied = replace(config, k0_decay_floor=rec.floor,
                          k0_decay_segment=rec.segment)
        result = SNNSolver(_stiff(), applied).solve(np.zeros(3))
        assert result.kkt_normalized <= TARGET
        assert result.decay_levels[-1] == pytest.approx(rec.floor)
        assert sum(result.decay_iterations) == config.max_iterations

    def test_probe_cost_is_reported(self):
        rec = recommend_k0_decay(_stiff(), SolverConfig(max_iterations=8000),
                                 target=TARGET)
        assert rec.evidence["probe_solves"] >= 2
        assert rec.evidence["probe_iterations_total"] > 0
        assert rec.evidence["verification_solves"] >= 1
        assert rec.evidence["verification_iterations_total"] > 0
        assert rec.evidence["wall_seconds"] >= 0.0
        assert len(rec.evidence["probes"]) >= 2
        assert len(rec.evidence["verifications"]) == rec.evidence["verification_solves"]
        for attempt in rec.evidence["verifications"]:
            assert attempt["iterations_used"] >= 0

    def test_verification_count_is_bounded(self):
        for allowed in (1, 2, 4):
            rec = recommend_k0_decay(
                _stiff(), SolverConfig(max_iterations=2000),
                target=TARGET, max_verifications=allowed)
            assert 1 <= len(rec.evidence["verifications"]) <= allowed

    def test_measurement_solves_consume_their_budget(self):
        # Early stopping is disabled for the measurement, so a probe never
        # stops at the caller's gate: comparing two steps only means
        # something when both reach their fixed point.
        rec = recommend_k0_decay(_stiff(), SolverConfig(max_iterations=2000),
                                 target=1e-6)
        for probe in rec.evidence["probes"]:
            assert probe["iterations_used"] == probe["iterations"]
            assert probe["converged"] is False

    def test_plateau_probe_runs_on_small_budgets(self):
        rec = recommend_k0_decay(_stiff(), SolverConfig(max_iterations=1000),
                                 target=1e-3)
        assert rec.evidence["plateau_ratio"] is not None
        assert rec.evidence["probes"][1]["iterations"] == 2000


class TestHealthyProblem:
    def test_recommends_off(self):
        rec = recommend_k0_decay(_healthy(), SolverConfig(), target=TARGET)
        assert rec.floor is None
        assert rec.levels == []
        assert rec.segment is None
        assert rec.reason == "target_already_met"
        assert rec.evidence["meets_target"] is False
        assert rec.evidence["probes"][0]["kkt_normalized"] <= TARGET


class TestFloorConstraints:
    def test_min_floor_is_respected_and_reported(self):
        config = SolverConfig(max_iterations=8000)
        baseline = recommend_k0_decay(_stiff(), config, target=TARGET)
        floor_floor = baseline.evidence["model_floor"] * 0.5
        min_floor = floor_floor * 4.0
        rec = recommend_k0_decay(_stiff(), config, target=TARGET,
                                 min_floor=min_floor)
        assert rec.floor is not None
        assert rec.floor >= min_floor
        assert rec.evidence["floor_clamped_to_min"] is True

    def test_min_floor_at_or_above_start_refuses(self):
        config = SolverConfig(max_iterations=8000)
        k0 = SNNSolver(_stiff(), config)._k0
        rec = recommend_k0_decay(_stiff(), config, target=TARGET,
                                 min_floor=2.0 * k0)
        assert rec.floor is None
        assert rec.reason in {"min_floor_at_or_above_start",
                              "no_usable_floor_within_budget"}


class TestContract:
    def test_config_is_not_mutated(self):
        config = SolverConfig(max_iterations=8000)
        before = asdict(config)
        recommend_k0_decay(_stiff(), config, target=TARGET)
        assert asdict(config) == before

    def test_none_path_stays_byte_identical_after_a_recommendation(self):
        # A genuine pre/post control: the reference is captured before the
        # recommender runs, and the post run must match it array for array.
        config = SolverConfig(max_iterations=2000, backend="python")
        reference = SNNSolver(_stiff(), config).solve(np.zeros(3))
        recommend_k0_decay(_stiff(), config, target=TARGET)
        after = SNNSolver(_stiff(), config).solve(np.zeros(3))
        assert np.array_equal(reference.final_x, after.final_x)
        assert reference.final_objective == after.final_objective
        assert reference.kkt_residual == after.kkt_residual
        assert after.decay_levels is None
        assert after.decay_iterations is None

    def test_rejects_non_euler(self):
        with pytest.raises(ValueError, match="euler"):
            recommend_k0_decay(_stiff(),
                               SolverConfig(integration_method="ivp"))

    def test_rejects_transform(self):
        with pytest.raises(ValueError, match="transformed"):
            recommend_k0_decay(_stiff(), SolverConfig(transform="eigenbasis"))

    def test_rejects_bad_target_and_margin(self):
        for kwargs in ({"target": 0.0}, {"target": float("nan")},
                       {"floor_margin": 0.0}, {"floor_margin": 1.5}):
            with pytest.raises(ValueError):
                recommend_k0_decay(_stiff(), SolverConfig(), **kwargs)


class _FakeResult:
    def __init__(self, normalized, residual=1.0, scale=1.0, fit="ok",
                 exhausted=False, levels=None, iterations_used=10,
                 converged=False):
        self.kkt_normalized = normalized
        self.kkt_residual = residual
        self.kkt_scale = scale
        self.kkt_fit_status = fit
        self.projection_budget_exhausted = exhausted
        self.decay_levels = levels
        self.iterations_used = iterations_used
        self.converged = converged
        self.convergence_reason = "max_iterations"


def _fake_solver(verification, probe=None):
    """A solver double whose probes show a clean O(k0) signature.

    The decision logic, not the dynamics, is what these tests pin: by
    default the probes return a residual proportional to k0, and
    ``verification`` supplies the staircase outcome for each candidate
    floor.
    """
    if probe is None:
        def probe(config):
            return _FakeResult(config.k0 / 0.5 * 0.02)

    class FakeSolver:
        def __init__(self, problem, config):
            self.config = config
            self._k0 = 0.5 if config.k0 is None else float(config.k0)

        def solve(self, x0, verbose=False):
            if self.config.k0_decay_floor is None:
                return probe(self.config)
            return verification(self.config)

    return FakeSolver


class TestVerificationSafety:
    def test_projection_abort_is_never_accepted(self, monkeypatch):
        def aborted(config):
            return _FakeResult(1e-9, residual=1e-9, exhausted=True,
                               levels=[0.5, config.k0_decay_floor])

        monkeypatch.setattr(recommend_module, "SNNSolver",
                            _fake_solver(aborted))
        rec = recommend_k0_decay(_stiff(), SolverConfig(max_iterations=8000),
                                 target=TARGET)
        assert rec.floor is None
        assert rec.reason == "verification_projection_budget_exhausted"
        assert all(v["projection_budget_exhausted"]
                   for v in rec.evidence["verifications"])

    def test_unreached_floor_is_never_accepted(self, monkeypatch):
        def stopped_early(config):
            return _FakeResult(1e-9, residual=1e-9, levels=[0.5])

        monkeypatch.setattr(recommend_module, "SNNSolver",
                            _fake_solver(stopped_early))
        rec = recommend_k0_decay(_stiff(), SolverConfig(max_iterations=8000),
                                 target=TARGET)
        assert rec.floor is None
        assert rec.reason == "no_usable_floor_within_budget"
        assert all(not v["reached_floor"]
                   for v in rec.evidence["verifications"])

    def test_a_clean_verification_is_accepted(self, monkeypatch):
        def clean(config):
            return _FakeResult(5e-5, residual=5e-5,
                               levels=[0.5, config.k0_decay_floor])

        monkeypatch.setattr(recommend_module, "SNNSolver",
                            _fake_solver(clean))
        rec = recommend_k0_decay(_stiff(), SolverConfig(max_iterations=8000),
                                 target=TARGET)
        assert rec.floor is not None
        assert rec.reason == "recommended"
        assert rec.evidence["meets_target"] is True
        assert len(rec.evidence["verifications"]) == 1

    def test_zero_scale_probe_reads_as_target_met(self, monkeypatch):
        def zero_scale(config):
            return _FakeResult(float("nan"), residual=0.0, scale=0.0)

        monkeypatch.setattr(recommend_module, "SNNSolver",
                            _fake_solver(zero_scale, probe=zero_scale))
        rec = recommend_k0_decay(_stiff(), SolverConfig(max_iterations=8000),
                                 target=TARGET)
        assert rec.floor is None
        assert rec.reason == "target_already_met"

    def test_zero_scale_verification_is_accepted(self, monkeypatch):
        # A verification that reaches an exact zero-residual/zero-scale
        # certificate is a pass, not missing evidence.
        def zero_scale(config):
            return _FakeResult(float("nan"), residual=0.0, scale=0.0,
                               levels=[0.5, config.k0_decay_floor])

        monkeypatch.setattr(recommend_module, "SNNSolver",
                            _fake_solver(zero_scale))
        rec = recommend_k0_decay(_stiff(), SolverConfig(max_iterations=8000),
                                 target=TARGET)
        assert rec.floor is not None
        assert rec.reason == "recommended"
        assert rec.evidence["meets_target"] is True
        assert np.isnan(rec.evidence["achieved_kkt_normalized"])

    def test_max_verifications_must_be_positive(self):
        with pytest.raises(ValueError, match="max_verifications"):
            recommend_k0_decay(_stiff(), SolverConfig(), max_verifications=0)


def _slow_diagonal(small_eigenvalue):
    """A QP whose slow coordinate makes more iterations the right fix."""
    return OptimizationProblem(
        A=np.diag([float(small_eigenvalue), 1.0]),
        b=np.array([-1.0, -1.0]),
        C=np.array([[-1.0, 0.0]]), d=np.array([-5.0]))


class TestRefusals:
    def test_slow_convergence_refuses(self):
        rec = recommend_k0_decay(_slow_diagonal(1e-2),
                                 SolverConfig(max_iterations=2000),
                                 target=1e-6)
        assert rec.floor is None
        assert rec.reason == "residual_improves_with_iterations"
        assert rec.evidence["plateau_ratio"] < 0.9

    def test_non_linear_step_refuses(self):
        rec = recommend_k0_decay(_slow_diagonal(1e-4),
                                 SolverConfig(max_iterations=2000),
                                 target=1e-6)
        assert rec.floor is None
        assert rec.reason == "residual_not_linear_in_k0"
        assert rec.evidence["k0_linearity_ratio"] > 0.7

    def test_budget_limited_recommendation_is_flagged(self):
        rec = recommend_k0_decay(_stiff(), SolverConfig(max_iterations=2000),
                                 target=TARGET)
        assert rec.floor is not None
        assert rec.reason == "recommended_above_target"
        assert rec.evidence["meets_target"] is False
        assert 1 <= len(rec.evidence["verifications"]) <= 3

    def test_plateau_check_can_be_disabled(self):
        rec = recommend_k0_decay(_stiff(), SolverConfig(max_iterations=8000),
                                 target=TARGET, check_plateau=False)
        assert len(rec.evidence["probes"]) == 2
        assert rec.evidence["plateau_ratio"] is None
        assert rec.reason == "recommended"

    def test_warm_start_is_used(self):
        rec = recommend_k0_decay(_stiff(), SolverConfig(max_iterations=8000),
                                 target=TARGET, x0=np.ones(3))
        assert rec.evidence["probe_start"] == "caller"
        assert rec.reason == "recommended"


class TestEvidenceSchema:
    def test_every_verification_attempt_has_the_same_keys(self):
        # A budget too small to reach the model floor appends a sentinel
        # attempt; it must carry the same keys as a real run so a consumer
        # can walk the list unconditionally.
        rec = recommend_k0_decay(_stiff(),
                                 SolverConfig(max_iterations=8),
                                 target=1e-4, probe_iterations=2000)
        expected = {
            "k0", "iterations", "iterations_used", "kkt_residual",
            "kkt_scale", "kkt_normalized", "kkt_fit_status", "converged",
            "convergence_reason", "projection_budget_exhausted", "floor",
            "n_levels", "segment", "reached_floor", "decay_levels",
            "meets_target",
        }
        assert rec.evidence["verifications"]
        for attempt in rec.evidence["verifications"]:
            assert set(attempt) == expected
            assert attempt["iterations_used"] >= 0


class TestCertificateHonesty:
    def test_failed_fit_probe_is_unavailable(self, monkeypatch):
        def failed(config):
            return _FakeResult(0.02, residual=0.02, scale=1.0,
                               fit="fit_failed")

        monkeypatch.setattr(recommend_module, "SNNSolver",
                            _fake_solver(failed, probe=failed))
        rec = recommend_k0_decay(_stiff(), SolverConfig(max_iterations=8000),
                                 target=TARGET)
        assert rec.floor is None
        assert rec.reason == "kkt_certificate_unavailable"

    def test_failed_fit_verification_is_not_accepted(self, monkeypatch):
        def failed(config):
            return _FakeResult(0.0, residual=0.0, scale=1.0, fit="fit_failed",
                               levels=[0.5, config.k0_decay_floor])

        monkeypatch.setattr(recommend_module, "SNNSolver",
                            _fake_solver(failed))
        rec = recommend_k0_decay(_stiff(), SolverConfig(max_iterations=8000),
                                 target=TARGET)
        assert rec.floor is None
        assert rec.reason == "no_usable_floor_within_budget"

    def test_a_tiny_floor_must_actually_be_executed(self, monkeypatch):
        # The last executed level is twice the candidate. Relative equality is
        # strict enough to reject it even though the absolute gap is tiny.
        def skipped_floor(config):
            candidate = float(config.k0_decay_floor)
            return _FakeResult(0.0, residual=0.0, scale=1.0,
                               levels=[0.5, 2.0 * candidate])

        monkeypatch.setattr(recommend_module, "SNNSolver",
                            _fake_solver(skipped_floor))
        rec = recommend_k0_decay(_stiff(), SolverConfig(max_iterations=8000),
                                 target=1e-11)
        assert rec.floor is None
        assert rec.reason == "no_usable_floor_within_budget"
        assert all(not v["reached_floor"]
                   for v in rec.evidence["verifications"])


class TestProbeCertificateGate:
    def test_failed_fit_on_the_halved_probe_is_unavailable(self, monkeypatch):
        def probe(config):
            if config.k0 < 0.4:
                # Only the half-k0 probe fails its fit; its diagnostic is
                # finite, so an ungated read would call it linearity evidence.
                return _FakeResult(0.01, residual=0.01, scale=1.0,
                                   fit="fit_failed")
            return _FakeResult(config.k0 / 0.5 * 0.02)

        def clean(config):
            return _FakeResult(5e-5, residual=5e-5,
                               levels=[0.5, config.k0_decay_floor])

        monkeypatch.setattr(recommend_module, "SNNSolver",
                            _fake_solver(clean, probe=probe))
        rec = recommend_k0_decay(_stiff(), SolverConfig(max_iterations=8000),
                                 target=TARGET)
        assert rec.floor is None
        assert rec.reason == "kkt_certificate_unavailable"
        assert len(rec.evidence["probes"]) == 3

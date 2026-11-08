"""Opt-in piecewise-constant geometric k0 schedule.

``k0_decay_floor is None`` (the default) is the constant-k0 path. The
reference arrays in ``tests/fixtures/k0_decay_offpath_reference.npz`` were
captured from that path before the schedule existed.
"""

from pathlib import Path

import numpy as np
import pytest

from snn_opt import OptimizationProblem, SNNSolver, SolverConfig
from snn_opt.solver import ConvergenceConfig

FIXTURE = Path(__file__).parent / "fixtures" / "k0_decay_offpath_reference.npz"

# Worked pair. Halve while the step is strictly above 1.5 * floor, then
# append the floor. That sequence has 13 levels. The next half sits in
# (floor, 1.5 * floor] and is not a level.
_K0_EFF = 0.2585
_FLOOR = 5.179e-5
_N_LEVELS = 13


def _problem(A, b, C, d):
    return OptimizationProblem(A=np.asarray(A, float), b=np.asarray(b, float),
                              C=np.asarray(C, float), d=np.asarray(d, float))


def _eye_problem(n=2):
    return _problem(np.eye(n), np.zeros(n), np.zeros((0, n)), np.zeros(0))


def _solve(problem, x0, **cfg):
    solver = SNNSolver(problem, SolverConfig(**cfg))
    return solver.solve(np.asarray(x0, float))


def _expect_levels(k0_eff, floor):
    """Levels the schedule rule produces.

    Start at ``k0_eff``. While the value is strictly above ``1.5 * floor``,
    keep it and halve. Then append ``floor`` exactly.
    """
    levels = []
    step = float(k0_eff)
    limit = 1.5 * float(floor)
    while step > limit:
        levels.append(step)
        step *= 0.5
    levels.append(float(floor))
    return levels


def _expect_counts(n_levels, max_iterations, segment):
    if segment is None:
        segment = max_iterations // n_levels
    counts = [int(segment)] * (n_levels - 1)
    counts.append(int(max_iterations - int(segment) * (n_levels - 1)))
    return counts


class TestLevelSchedule:
    def test_worked_pair_level_count_and_values(self):
        levels = _expect_levels(_K0_EFF, _FLOOR)
        assert len(levels) == _N_LEVELS
        # The penultimate level is the last halving still above 1.5 * floor.
        # The next half is above the floor and is not given its own level.
        assert levels[-2] > 1.5 * _FLOOR
        skipped = levels[-2] * 0.5
        assert _FLOOR < skipped <= 1.5 * _FLOOR
        assert all(abs(level - skipped) > 0.0 for level in levels)
        problem = _eye_problem()
        result = _solve(
            problem, np.zeros(2),
            k0=_K0_EFF, k0_decay_floor=_FLOOR,
            max_iterations=_N_LEVELS, record_trajectory=False,
            backend="python",
            convergence=ConvergenceConfig(enable_early_stopping=False),
        )
        assert result.decay_levels is not None
        assert len(result.decay_levels) == _N_LEVELS
        assert result.decay_levels[-1] == _FLOOR
        for got, exp in zip(result.decay_levels, levels):
            assert got == pytest.approx(exp, rel=0, abs=0)
        assert result.decay_iterations == [1] * _N_LEVELS
        assert result.iterations_used == _N_LEVELS

    def test_auto_segment_puts_remainder_on_the_final_level(self):
        # ratio 8 = 2^3 -> 4 levels: 8, 4, 2, 1. 102 // 4 = 25, remainder 2
        # on the floor level -> 25, 25, 25, 27.
        max_iterations = 102
        result = _solve(
            _eye_problem(), np.ones(2),
            k0=8.0, k0_decay_floor=1.0,
            max_iterations=max_iterations, record_trajectory=False,
            backend="python",
            convergence=ConvergenceConfig(enable_early_stopping=False),
        )
        expected = _expect_counts(4, max_iterations, None)
        assert expected == [25, 25, 25, 27]
        assert result.decay_iterations == expected
        assert sum(result.decay_iterations) == max_iterations
        assert result.iterations_used == max_iterations
        assert result.decay_levels == pytest.approx([8.0, 4.0, 2.0, 1.0])

    def test_explicit_segment_overrides_the_split(self):
        max_iterations = 100
        result = _solve(
            _eye_problem(), np.ones(2),
            k0=8.0, k0_decay_floor=1.0, k0_decay_segment=10,
            max_iterations=max_iterations, record_trajectory=False,
            backend="python",
            convergence=ConvergenceConfig(enable_early_stopping=False),
        )
        assert result.decay_iterations == [10, 10, 10, 70]
        assert sum(result.decay_iterations) == max_iterations
        assert result.iterations_used == max_iterations


class TestValidation:
    def _run(self, **cfg):
        _solve(_eye_problem(), np.zeros(2),
               max_iterations=cfg.pop("max_iterations", 50),
               record_trajectory=False, backend="python", **cfg)

    def test_floor_must_be_positive_finite_and_below_k0_eff(self):
        for floor in (0.0, -1.0, float("nan"), float("inf"), float("-inf")):
            with pytest.raises(ValueError, match="k0_decay_floor"):
                self._run(k0=1.0, k0_decay_floor=floor)
        with pytest.raises(ValueError, match="k0_decay_floor"):
            self._run(k0=0.5, k0_decay_floor=0.5)
        with pytest.raises(ValueError, match="k0_decay_floor"):
            self._run(k0=0.5, k0_decay_floor=1.0)

    def test_segment_requires_a_floor_and_must_be_a_positive_int(self):
        with pytest.raises(ValueError, match="k0_decay_segment"):
            self._run(k0=1.0, k0_decay_segment=4)
        with pytest.raises(ValueError, match="k0_decay_segment"):
            self._run(k0=1.0, k0_decay_floor=0.2, k0_decay_segment=0)
        with pytest.raises(ValueError, match="k0_decay_segment"):
            self._run(k0=1.0, k0_decay_floor=0.2, k0_decay_segment=-3)
        with pytest.raises(ValueError, match="k0_decay_segment"):
            self._run(k0=1.0, k0_decay_floor=0.2, k0_decay_segment=True)

    def test_budget_must_reach_the_floor(self):
        with pytest.raises(ValueError, match="k0_decay_floor"):
            self._run(k0=_K0_EFF, k0_decay_floor=_FLOOR, max_iterations=10)
        # 13 levels * segment 2 = 26 > 20.
        with pytest.raises(ValueError, match="k0_decay_segment"):
            self._run(k0=_K0_EFF, k0_decay_floor=_FLOOR,
                      k0_decay_segment=2, max_iterations=20)

    def test_adaptive_k0_eff_rejects_a_floor_at_or_above_it(self):
        # Identity Hessian, k0_scale 0.5 -> k0_eff = 0.5.
        with pytest.raises(ValueError, match="k0_decay_floor"):
            self._run(k0=None, k0_scale=0.5, k0_decay_floor=0.5)

    def test_ivp_is_rejected(self):
        # The IVP path integrates to t_end and ignores the per-level
        # max_iterations, so a staircase there would have no budget.
        with pytest.raises(ValueError, match="integration_method"):
            self._run(k0=1.0, k0_decay_floor=0.2,
                      integration_method="ivp", t_end=0.1, max_step=0.01)


class TestGateAndAbort:
    def test_nonfinal_levels_run_the_full_allocation_final_level_can_stop(self):
        # Already at the unconstrained optimum. A live gate stops in a
        # couple of iterations (see the floor level). The first two levels
        # have the gate off, so each still consumes its full allocation.
        # ratio 4 -> 3 levels, 90 iterations -> 30 each.
        convergence = ConvergenceConfig(
            enable_early_stopping=True,
            min_iterations=0,
            check_every=1,
            patience=1,
            window_size=2,
            obj_rel_tol=1e-8,
            feasibility_tol=1e-6,
        )
        result = _solve(
            _problem(4.0 * np.eye(2), np.zeros(2), np.zeros((0, 2)), np.zeros(0)),
            np.zeros(2),
            k0=0.2, k0_decay_floor=0.05,
            max_iterations=90, record_trajectory=False, backend="python",
            convergence=convergence,
        )
        assert result.decay_iterations[:2] == [30, 30]
        assert result.decay_iterations[2] < 30
        assert result.converged
        assert result.iterations_used == sum(result.decay_iterations)
        assert result.convergence_reason != "projection_budget_exhausted"

    def test_projection_abort_in_a_nonfinal_level_ends_the_schedule(self):
        rng = np.random.default_rng(0)
        n = 20
        factor = rng.standard_normal((n, 6))
        A = factor @ factor.T / 6.0 + np.diag(0.1 + 0.2 * rng.random(n))
        b = -(0.05 + 0.15 * rng.random(n)) / 5.0
        C = np.ones((1, n))
        d = np.array([-1.0])
        result = _solve(
            _problem(A, b, C, d), np.zeros(n),
            k0=1.0, k0_decay_floor=0.4,
            max_iterations=30, max_projection_iters=3,
            lower_bound=0.0, upper_bound=1.0,
            record_trajectory=False, backend="python",
        )
        assert result.projection_budget_exhausted
        assert result.convergence_reason == "projection_budget_exhausted"
        assert result.decay_levels is not None
        assert len(result.decay_levels) == 1
        assert len(result.decay_iterations) == 1
        assert not result.converged
        assert result.iterations_used == result.decay_iterations[0]


class TestWatchdog:
    def test_decay_raises_an_unspecified_cap_to_at_least_5000(self):
        bare = SNNSolver(_eye_problem(), SolverConfig())
        assert bare._proj_cap == 1000
        raised = SNNSolver(
            _eye_problem(),
            SolverConfig(k0=1.0, k0_decay_floor=0.25),
        )
        assert raised._proj_cap == 5000

    def test_decay_keeps_a_larger_automatic_cap(self):
        n, m = 2, 600
        problem = _problem(np.eye(n), np.zeros(n), np.ones((m, n)), -np.ones(m))
        plain = SNNSolver(problem, SolverConfig())
        # max(1000, 10 * m) = 6000
        assert plain._proj_cap == 6000
        raised = SNNSolver(
            problem, SolverConfig(k0=1.0, k0_decay_floor=0.25),
        )
        assert raised._proj_cap == 6000

    def test_explicit_cap_is_not_rewritten(self):
        solver = SNNSolver(
            _eye_problem(),
            SolverConfig(k0=1.0, k0_decay_floor=0.25, max_projection_iters=1000),
        )
        assert solver._proj_cap == 1000


class TestOffPath:
    def test_floor_none_matches_the_prechange_solution(self):
        data = np.load(FIXTURE)
        problem = _problem(data["A"], data["b"], data["C"], data["d"])
        x0 = np.asarray(data["x0"], float)
        lean = _solve(
            problem, x0,
            max_iterations=int(data["max_iterations"]),
            k0_scale=float(data["k0_scale"]),
            constraint_tol=float(data["constraint_tol"]),
            record_trajectory=False,
            backend="python",
            projection_method="adaptive",
            k0_decay_floor=None,
        )
        assert np.array_equal(lean.final_x, data["final_x"])
        assert lean.final_objective == float(data["final_objective"])
        assert lean.iterations_used == int(data["iterations_used"])
        assert lean.decay_levels is None
        assert lean.decay_iterations is None

        recorded = _solve(
            problem, x0,
            max_iterations=int(data["recorded_max_iterations"]),
            k0_scale=float(data["k0_scale"]),
            constraint_tol=float(data["constraint_tol"]),
            record_trajectory=True,
            record_spike_history=False,
            backend="python",
            projection_method="adaptive",
        )
        assert np.array_equal(recorded.final_x, data["recorded_final_x"])
        assert recorded.iterations_used == int(data["recorded_iterations_used"])
        assert recorded.decay_levels is None


def _box_problem():
    """Identity objective with a lower bound, so every step projects."""
    return _problem(np.eye(2), np.zeros(2), np.zeros((0, 2)), np.zeros(0))


def _replay_box_phases(levels, counts, *, record_trajectory, observe, cap):
    """Run the same levels the schedule would, as separate solves."""
    convergence = ConvergenceConfig(enable_early_stopping=False)
    x = np.zeros(2)
    phases = []
    for level, n_iter in zip(levels, counts):
        phase = _solve(
            _box_problem(), x,
            k0=float(level),
            max_iterations=int(n_iter),
            max_projection_iters=cap,
            lower_bound=1.0,
            record_trajectory=record_trajectory,
            record_spike_history=True,
            observe_projection_events=observe,
            backend="python",
            convergence=convergence,
        )
        phases.append(phase)
        x = np.asarray(phase.final_x, float).copy()
    return phases


class TestTrajectoryAndWork:
    def test_recorded_trajectory_concatenates_every_level(self):
        # k0 0.5, floor 0.2 -> two levels. Four iterations -> 2 + 2.
        convergence = ConvergenceConfig(enable_early_stopping=False)
        x0 = np.zeros(2)
        result = _solve(
            _box_problem(), x0,
            k0=0.5, k0_decay_floor=0.2,
            max_iterations=4,
            lower_bound=1.0,
            record_trajectory=True,
            record_spike_history=True,
            backend="python",
            convergence=convergence,
        )
        assert result.decay_iterations == [2, 2]
        assert result.iterations_used == 4
        phases = _replay_box_phases(
            result.decay_levels, result.decay_iterations,
            record_trajectory=True, observe=False, cap=5000)
        assert len(phases) == 2
        # Later phases drop the handoff row, so the series matches one
        # recorded solve: the original x0 plus one row per iteration.
        assert result.t.shape == (result.iterations_used + 1,)
        assert np.array_equal(result.t, np.arange(result.iterations_used + 1, dtype=float))
        assert result.X.shape == (result.iterations_used + 1, 2)
        assert result.objective_values.shape == result.t.shape
        assert result.constraint_violations.shape == result.t.shape
        assert np.array_equal(result.X[0], x0)
        assert np.array_equal(result.X[-1], result.final_x)
        offset = 0
        t_parts = [np.asarray(phases[0].t, float)]
        x_parts = [np.asarray(phases[0].X, float)]
        obj_parts = [np.asarray(phases[0].objective_values, float)]
        viol_parts = [np.asarray(phases[0].constraint_violations, float)]
        spike_times = [np.asarray(phases[0].spike_times, float)]
        spike_deltas = [np.asarray(phases[0].spike_deltas, float)]
        spike_norms = [np.asarray(phases[0].spike_norms, float)]
        constraints = list(phases[0].spike_constraints)
        violations = list(phases[0].spike_violation_values)
        offset = int(phases[0].iterations_used)
        for phase in phases[1:]:
            t_parts.append(np.asarray(phase.t, float).reshape(-1)[1:] + offset)
            x_parts.append(np.asarray(phase.X, float)[1:])
            obj_parts.append(np.asarray(phase.objective_values, float).reshape(-1)[1:])
            viol_parts.append(np.asarray(phase.constraint_violations, float).reshape(-1)[1:])
            spike_times.append(np.asarray(phase.spike_times, float) + offset)
            spike_deltas.append(np.asarray(phase.spike_deltas, float))
            spike_norms.append(np.asarray(phase.spike_norms, float))
            constraints.extend(phase.spike_constraints)
            violations.extend(phase.spike_violation_values)
            offset += int(phase.iterations_used)
        assert np.array_equal(result.t, np.concatenate(t_parts))
        assert np.array_equal(result.X, np.vstack(x_parts))
        assert np.allclose(result.objective_values, np.concatenate(obj_parts))
        assert np.allclose(result.constraint_violations, np.concatenate(viol_parts))
        assert np.array_equal(result.spike_times, np.concatenate(spike_times))
        assert np.array_equal(result.spike_deltas, np.vstack(spike_deltas))
        assert np.allclose(result.spike_norms, np.concatenate(spike_norms))
        assert len(result.spike_constraints) == len(constraints)
        assert len(result.spike_violation_values) == len(violations)
        assert result.spike_times.size > 0
        assert float(result.spike_times.max()) >= phases[0].iterations_used

    def test_work_counters_sum_across_levels(self):
        convergence = ConvergenceConfig(enable_early_stopping=False)
        result = _solve(
            _box_problem(), np.zeros(2),
            k0=0.5, k0_decay_floor=0.2,
            max_iterations=4,
            lower_bound=1.0,
            record_trajectory=True,
            record_spike_history=True,
            observe_projection_events=True,
            backend="python",
            convergence=convergence,
        )
        phases = _replay_box_phases(
            result.decay_levels, result.decay_iterations,
            record_trajectory=True, observe=True, cap=5000)
        assert len(phases) >= 2
        assert result.n_projections == sum(int(phase.n_projections) for phase in phases)
        assert result.n_projections > int(phases[-1].n_projections)
        assert result.total_projection_distance == pytest.approx(
            sum(float(phase.total_projection_distance) for phase in phases))
        assert result.total_projection_distance > float(phases[-1].total_projection_distance)
        assert result.projection_truncated_sweeps == sum(
            int(phase.projection_truncated_sweeps) for phase in phases)
        assert result.dykstra_inner_cap_hits == sum(
            int(phase.dykstra_inner_cap_hits) for phase in phases)
        assert result.dykstra_inner_iterations.shape[0] == sum(
            np.asarray(phase.dykstra_inner_iterations).shape[0] for phase in phases)
        assert result.observed_total_projection_distance == pytest.approx(
            sum(float(phase.observed_total_projection_distance) for phase in phases))
        assert result.projection_cap_rechecks == sum(
            int(phase.projection_cap_rechecks) for phase in phases)
        assert np.array_equal(
            result.implicit_lower_event_counts,
            sum(np.asarray(phase.implicit_lower_event_counts) for phase in phases))
        assert result.projection_event_digest == phases[-1].projection_event_digest
        assert result.projection_event_digest != (
            int(phases[0].projection_event_digest) + int(phases[-1].projection_event_digest))
        assert result.projection_first_candidate_id == phases[0].projection_first_candidate_id
        assert result.projection_last_candidate_id == phases[-1].projection_last_candidate_id

        lean = _solve(
            _box_problem(), np.zeros(2),
            k0=0.5, k0_decay_floor=0.2,
            max_iterations=4,
            lower_bound=1.0,
            record_trajectory=False,
            backend="python",
            convergence=convergence,
        )
        assert lean.n_projections == result.n_projections
        assert lean.total_projection_distance == 0.0
        assert lean.t.shape == (1,)
        assert lean.t[0] == lean.iterations_used


class TestCapResolvedAtSolve:
    def test_floor_set_after_construction_raises_the_cap_at_solve(self):
        solver = SNNSolver(
            _eye_problem(),
            SolverConfig(k0=1.0, max_iterations=6, record_trajectory=False,
                         backend="python"),
        )
        assert solver._proj_cap == 1000
        solver.config.k0_decay_floor = 0.25
        solver.solve(np.zeros(2))
        assert solver._proj_cap == 5000

    def test_explicit_cap_stays_when_the_floor_is_set_after_construction(self):
        solver = SNNSolver(
            _eye_problem(),
            SolverConfig(k0=1.0, max_iterations=6, max_projection_iters=1000,
                         record_trajectory=False, backend="python"),
        )
        assert solver._proj_cap == 1000
        solver.config.k0_decay_floor = 0.25
        solver.solve(np.zeros(2))
        assert solver._proj_cap == 1000

    def test_constant_path_does_not_recompute_the_cap(self):
        solver = SNNSolver(
            _eye_problem(),
            SolverConfig(max_iterations=2, record_trajectory=False, backend="python"),
        )
        assert solver._proj_cap == 1000
        solver.solve(np.zeros(2))
        assert solver._proj_cap == 1000


class TestBackends:
    def test_c_backend_runs_the_schedule(self):
        result = _solve(
            _eye_problem(), np.zeros(2),
            k0=1.0, k0_decay_floor=0.2, max_iterations=40,
            record_trajectory=False, backend="c",
        )
        assert result.decay_levels == pytest.approx([1.0, 0.5, 0.2])
        assert result.iterations_used == 40

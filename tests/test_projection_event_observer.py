"""Observation-only projection-event counter and digest fixtures.

All numerical inputs are integers or dyadic values. This makes exact
counter-off/on and Python/C comparisons meaningful, while the broader native
parity suite retains its existing tolerance policy for BLAS versus C loops.
"""

import numpy as np
import pytest

from snn_opt import (ConvergenceConfig, OptimizationProblem, SNNSolver,
                     SolverConfig)


pytest.importorskip(
    "snn_opt._kernel",
    reason="compiled C++ kernel not built (python setup.py build_ext --inplace)",
)


_DIGEST_OFFSET = 14695981039346656037
_DIGEST_PRIME = 1099511628211
_UINT64_MASK = (1 << 64) - 1
_BACKENDS = ("python", "c_serial")


def _digest(event_tokens):
    value = _DIGEST_OFFSET
    for token in event_tokens:
        for word in token:
            value = ((value ^ (int(word) + 1))
                     * _DIGEST_PRIME) & _UINT64_MASK
    return value


def _solve(case, backend, observe, *, record_trajectory=False):
    x0 = np.asarray(case["x0"], dtype=float)
    n = x0.size
    A = np.asarray(case.get("A", np.zeros((n, n))), dtype=float)
    b = np.asarray(case.get("b", np.zeros(n)), dtype=float)
    problem = OptimizationProblem(
        A=A,
        b=b,
        C=np.asarray(case["C"], dtype=float).reshape(-1, n),
        d=np.asarray(case["d"], dtype=float),
    )
    config = SolverConfig(
        k0=case.get("k0", 0.0),
        constraint_tol=1e-12,
        max_projection_iters=case.get("cap", 8),
        max_iterations=case.get("max_iterations", 1),
        lower_bound=case.get("lower"),
        upper_bound=case.get("upper"),
        backend=backend,
        record_trajectory=record_trajectory,
        record_spike_history=False,
        observe_projection_events=observe,
        convergence=ConvergenceConfig(enable_early_stopping=False),
    )
    return SNNSolver(problem, config).solve(x0)


_CASES = (
    pytest.param(
        {
            "C": [[1.0, 0.0]],
            "d": [-2.0],
            "x0": [0.0, 0.0],
            "final": [0.0, 0.0],
            "row": [0],
            "lower_counts": [0, 0],
            "upper_counts": [0, 0],
            "ids": [],
            "distance": 0.0,
        },
        id="no-event",
    ),
    pytest.param(
        {
            "C": [[1.0, 0.0]],
            "d": [0.0],
            "x0": [1.0, 0.0],
            "final": [0.0, 0.0],
            "row": [1],
            "lower_counts": [0, 0],
            "upper_counts": [0, 0],
            "ids": [0],
            "distance": 1.0,
        },
        id="explicit-row",
    ),
    pytest.param(
        {
            "C": np.empty((0, 2)),
            "d": [],
            "x0": [-1.0, 0.0],
            "final": [0.0, 0.0],
            "lower": 0.0,
            "row": [],
            "lower_counts": [1, 0],
            "upper_counts": [0, 0],
            "ids": [0],
            "distance": 1.0,
        },
        id="implicit-lower",
    ),
    pytest.param(
        {
            "C": np.empty((0, 2)),
            "d": [],
            "x0": [2.0, 0.0],
            "final": [1.0, 0.0],
            "upper": 1.0,
            "row": [],
            "lower_counts": [0, 0],
            "upper_counts": [1, 0],
            "ids": [2],
            "distance": 1.0,
        },
        id="implicit-upper",
    ),
    pytest.param(
        {
            "C": [[0.0, 1.0]],
            "d": [-2.0],
            "x0": [2.0, 0.0],
            "final": [1.0, 0.0],
            "upper": 1.0,
            "row": [0],
            "lower_counts": [0, 0],
            "upper_counts": [1, 0],
            "ids": [3],
            "distance": 1.0,
        },
        id="implicit-upper-with-explicit-family",
    ),
    pytest.param(
        {
            "C": [[0.0, 1.0]],
            "d": [0.0],
            "x0": [-1.0, 1.0],
            "final": [0.0, 0.0],
            "lower": 0.0,
            "row": [1],
            "lower_counts": [1, 0],
            "upper_counts": [0, 0],
            "ids": [0, 1],
            "distance": 2.0,
        },
        id="row-lower-exact-tie",
    ),
    pytest.param(
        {
            "C": np.empty((0, 4)),
            "d": [],
            "x0": [-2.0, 0.25, 2.0, -3.0],
            "final": [0.0, 0.25, 1.0, 0.0],
            "lower": 0.0,
            "upper": 1.0,
            "row": [],
            "lower_counts": [1, 0, 0, 1],
            "upper_counts": [0, 0, 1, 0],
            # Box projection is simultaneous; observer order is canonicalized
            # to lower coordinates, then upper coordinates.
            "ids": [0, 3, 6],
            "distance": 6.0,
        },
        id="box-only-multi-coordinate",
    ),
    pytest.param(
        {
            "C": [[0.0, 1.0]],
            "d": [0.0],
            "x0": [-1.0, 1.0],
            "final": [-1.0, 0.0],
            "lower": 0.0,
            "cap": 1,
            "row": [1],
            "lower_counts": [0, 0],
            "upper_counts": [0, 0],
            "ids": [0],
            "distance": 1.0,
            "cap_rechecks": 1,
            "budget_exhausted": True,
        },
        id="projection-cap-exhaustion",
    ),
    pytest.param(
        {
            "A": np.zeros((1, 1)),
            "b": [-1.0],
            "k0": 1.0,
            "max_iterations": 4,
            "C": [[1.0]],
            "d": [0.0],
            "x0": [0.0],
            "final": [0.0],
            "cap": 1,
            "row": [4],
            "lower_counts": [0],
            "upper_counts": [0],
            "ids": [0, 0, 0, 0],
            "distance": 4.0,
            "tokens": [(outer_iteration, 0, 0)
                       for outer_iteration in range(4)],
            # Every one-event sweep consumes the cap, then its fresh
            # joint-violation recheck passes. This must not be conflated with
            # terminal projection-budget exhaustion.
            "cap_rechecks": 4,
            "budget_exhausted": False,
        },
        id="multiple-passing-projection-cap-rechecks",
    ),
    pytest.param(
        {
            "A": np.eye(2),
            "b": [-2.0, 2.0],
            "k0": 0.5,
            "max_iterations": 8,
            "C": np.empty((0, 2)),
            "d": [],
            "x0": [0.0, 0.0],
            "final": [1.0, 0.0],
            "lower": 0.0,
            "upper": 1.0,
            "row": [],
            "lower_counts": [0, 8],
            "upper_counts": [7, 0],
            "ids": [1] + [candidate for _ in range(7) for candidate in (1, 2)],
            "distance": 11.5,
            "tokens": (
                [(0, 0, 1)]
                + [token
                   for outer_iteration in range(1, 8)
                   for token in (
                       (outer_iteration, 0, 1),
                       (outer_iteration, 1, 2),
                   )]
            ),
        },
        id="multi-iteration-outer-ordinal-stream",
    ),
)


def _assert_observer(case, result):
    row = np.asarray(case["row"], dtype=np.int64)
    lower = np.asarray(case["lower_counts"], dtype=np.int64)
    upper = np.asarray(case["upper_counts"], dtype=np.int64)
    ids = case["ids"]
    tokens = case.get(
        "tokens",
        [(0, ordinal, candidate_id)
         for ordinal, candidate_id in enumerate(ids)],
    )

    np.testing.assert_array_equal(result.explicit_row_event_counts, row)
    np.testing.assert_array_equal(result.implicit_lower_event_counts, lower)
    np.testing.assert_array_equal(result.implicit_upper_event_counts, upper)
    assert result.explicit_row_events == int(row.sum())
    assert result.implicit_lower_events == int(lower.sum())
    assert result.implicit_upper_events == int(upper.sum())
    assert (result.explicit_row_events
            + result.implicit_lower_events
            + result.implicit_upper_events) == result.n_projections
    assert result.projection_event_digest == _digest(tokens)
    assert result.projection_event_digest_algorithm == "fnv1a64-word-v2"
    assert result.observed_total_projection_distance == case["distance"]
    assert result.projection_first_candidate_id == (ids[0] if ids else None)
    assert result.projection_last_candidate_id == (ids[-1] if ids else None)
    assert result.projection_cap_rechecks == case.get("cap_rechecks", 0)


@pytest.mark.parametrize("case", _CASES)
@pytest.mark.parametrize("backend", _BACKENDS)
def test_observer_is_recurrence_identity_within_backend(case, backend):
    plain = _solve(case, backend, observe=False)
    observed = _solve(case, backend, observe=True)

    assert np.array_equal(plain.final_x, observed.final_x)
    assert plain.final_objective == observed.final_objective
    assert plain.converged == observed.converged
    assert plain.convergence_reason == observed.convergence_reason
    assert plain.projection_budget_exhausted == observed.projection_budget_exhausted
    assert plain.iterations_used == observed.iterations_used
    assert plain.n_projections == observed.n_projections
    assert plain.explicit_row_event_counts is None
    assert plain.implicit_lower_event_counts is None
    assert plain.implicit_upper_event_counts is None
    assert plain.explicit_row_events is None
    assert plain.implicit_lower_events is None
    assert plain.implicit_upper_events is None
    assert plain.projection_event_digest is None
    assert plain.projection_event_digest_algorithm is None
    assert plain.observed_total_projection_distance is None
    assert plain.projection_first_candidate_id is None
    assert plain.projection_last_candidate_id is None
    assert plain.projection_cap_rechecks is None

    np.testing.assert_array_equal(observed.final_x, np.asarray(case["final"]))
    assert observed.projection_budget_exhausted == case.get("budget_exhausted", False)
    _assert_observer(case, observed)


@pytest.mark.parametrize("case", _CASES)
def test_observed_python_and_c_match_exactly_on_dyadic_fixtures(case):
    py = _solve(case, "python", observe=True)
    c = _solve(case, "c_serial", observe=True)

    assert np.array_equal(py.final_x, c.final_x)
    assert py.converged == c.converged
    assert py.projection_budget_exhausted == c.projection_budget_exhausted
    assert py.iterations_used == c.iterations_used
    assert py.n_projections == c.n_projections
    np.testing.assert_array_equal(
        py.explicit_row_event_counts, c.explicit_row_event_counts)
    np.testing.assert_array_equal(
        py.implicit_lower_event_counts, c.implicit_lower_event_counts)
    np.testing.assert_array_equal(
        py.implicit_upper_event_counts, c.implicit_upper_event_counts)
    assert py.projection_event_digest == c.projection_event_digest
    assert py.projection_event_digest_algorithm == c.projection_event_digest_algorithm
    assert (py.observed_total_projection_distance
            == c.observed_total_projection_distance)
    assert py.projection_first_candidate_id == c.projection_first_candidate_id
    assert py.projection_last_candidate_id == c.projection_last_candidate_id
    assert py.projection_cap_rechecks == c.projection_cap_rechecks


def test_python_full_and_lean_paths_report_the_same_observer_stream():
    case = {
        "C": [[0.0, 1.0]],
        "d": [0.0],
        "x0": [-1.0, 1.0],
        "lower": 0.0,
        "row": [1],
        "lower_counts": [1, 0],
        "upper_counts": [0, 0],
        "ids": [0, 1],
    }
    lean = _solve(case, "python", observe=True, record_trajectory=False)
    full = _solve(case, "python", observe=True, record_trajectory=True)

    assert np.array_equal(lean.final_x, full.final_x)
    assert lean.n_projections == full.n_projections
    np.testing.assert_array_equal(
        lean.explicit_row_event_counts, full.explicit_row_event_counts)
    np.testing.assert_array_equal(
        lean.implicit_lower_event_counts, full.implicit_lower_event_counts)
    np.testing.assert_array_equal(
        lean.implicit_upper_event_counts, full.implicit_upper_event_counts)
    assert lean.projection_event_digest == full.projection_event_digest
    assert (lean.observed_total_projection_distance
            == full.observed_total_projection_distance == 2.0)
    assert lean.projection_first_candidate_id == full.projection_first_candidate_id
    assert lean.projection_last_candidate_id == full.projection_last_candidate_id
    assert lean.projection_cap_rechecks == full.projection_cap_rechecks == 0


@pytest.mark.parametrize("backend", _BACKENDS)
def test_full_and_lean_paths_count_passing_cap_rechecks_without_terminal_flag(
        backend):
    case = {
        "A": np.zeros((1, 1)),
        "b": [-1.0],
        "k0": 1.0,
        "max_iterations": 4,
        "C": [[1.0]],
        "d": [0.0],
        "x0": [0.0],
        "cap": 1,
        "row": [4],
        "lower_counts": [0],
        "upper_counts": [0],
        "ids": [0, 0, 0, 0],
        "distance": 4.0,
        "tokens": [(outer_iteration, 0, 0)
                   for outer_iteration in range(4)],
        "cap_rechecks": 4,
    }
    lean = _solve(case, backend, observe=True, record_trajectory=False)
    full = _solve(case, backend, observe=True, record_trajectory=True)
    plain_lean = _solve(case, backend, observe=False, record_trajectory=False)
    plain_full = _solve(case, backend, observe=False, record_trajectory=True)

    assert np.array_equal(lean.final_x, full.final_x)
    assert lean.projection_budget_exhausted is False
    assert full.projection_budget_exhausted is False
    assert lean.convergence_reason == full.convergence_reason == "max_iterations"
    assert lean.n_projections == full.n_projections == 4
    assert lean.projection_cap_rechecks == full.projection_cap_rechecks == 4
    assert plain_lean.projection_cap_rechecks is None
    assert plain_full.projection_cap_rechecks is None
    _assert_observer(case, lean)
    _assert_observer(case, full)


def test_real_valued_projection_distance_parity_and_recurrence_identity():
    case = {
        "C": [[1.2, -0.7]],
        "d": [0.3],
        "x0": [0.9, -0.4],
        "row": [1],
        "lower_counts": [0, 0],
        "upper_counts": [0, 0],
        "ids": [0],
    }
    raw_violation = 1.2 * 0.9 + (-0.7) * (-0.4) + 0.3
    expected_distance = raw_violation / np.sqrt(1.2 ** 2 + (-0.7) ** 2)
    observed = {}

    for backend in _BACKENDS:
        plain = _solve(case, backend, observe=False)
        result = _solve(case, backend, observe=True)
        assert np.array_equal(plain.final_x, result.final_x)
        assert plain.iterations_used == result.iterations_used
        assert plain.n_projections == result.n_projections == 1
        assert plain.projection_budget_exhausted == result.projection_budget_exhausted
        assert plain.observed_total_projection_distance is None
        assert result.total_projection_distance == 0.0
        assert result.observed_total_projection_distance == pytest.approx(
            expected_distance, rel=1e-15, abs=1e-15)
        observed[backend] = result

    assert np.allclose(
        observed["python"].final_x,
        observed["c_serial"].final_x,
        rtol=0.0,
        atol=2e-16,
    )
    assert observed["python"].observed_total_projection_distance == pytest.approx(
        observed["c_serial"].observed_total_projection_distance,
        rel=1e-15,
        abs=1e-15,
    )


def _semantic_counts(result, explicit_class_codes):
    counts = np.zeros(5, dtype=np.int64)
    for count, class_code in zip(
            result.explicit_row_event_counts, explicit_class_codes):
        counts[int(class_code)] += int(count)
    counts[4] += result.implicit_lower_events
    return counts


@pytest.mark.parametrize("backend", _BACKENDS)
def test_full_row_and_hybrid_have_aligned_candidate_stream(backend):
    hybrid = {
        "C": [[0.0, 1.0]],
        "d": [0.0],
        "x0": [-1.0, 1.0],
        "lower": 0.0,
        "row": [1],
        "lower_counts": [1, 0],
        "upper_counts": [0, 0],
        "ids": [0, 1],
    }
    full_row = {
        "C": [[0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]],
        "d": [0.0, 0.0, 0.0],
        "x0": [-1.0, 1.0],
        "row": [1, 1, 0],
        "lower_counts": [0, 0],
        "upper_counts": [0, 0],
        "ids": [0, 1],
    }

    hybrid_result = _solve(hybrid, backend, observe=True)
    full_result = _solve(full_row, backend, observe=True)

    assert np.array_equal(hybrid_result.final_x, full_result.final_x)
    assert hybrid_result.iterations_used == full_result.iterations_used == 1
    assert hybrid_result.n_projections == full_result.n_projections == 2
    assert hybrid_result.projection_event_digest == full_result.projection_event_digest
    assert hybrid_result.projection_event_digest == _digest(
        [(0, 0, 0), (0, 1, 1)])
    assert hybrid_result.projection_event_digest_algorithm == "fnv1a64-word-v2"
    assert (hybrid_result.observed_total_projection_distance
            == full_result.observed_total_projection_distance == 2.0)
    assert hybrid_result.projection_first_candidate_id == 0
    assert hybrid_result.projection_last_candidate_id == 1
    assert full_result.projection_first_candidate_id == 0
    assert full_result.projection_last_candidate_id == 1

    hybrid_semantic = _semantic_counts(hybrid_result, [0])
    full_semantic = _semantic_counts(full_result, [0, 4, 4])
    np.testing.assert_array_equal(hybrid_semantic, [1, 0, 0, 0, 1])
    np.testing.assert_array_equal(full_semantic, hybrid_semantic)
    assert int(hybrid_semantic.sum()) == hybrid_result.n_projections
    assert int(full_semantic.sum()) == full_result.n_projections
    assert full_result.explicit_row_events == (
        hybrid_result.explicit_row_events + hybrid_result.implicit_lower_events)
    assert full_result.implicit_lower_events == 0
    assert full_result.implicit_upper_events == 0


def test_observer_rejects_fixed_projection():
    problem = OptimizationProblem(
        A=np.eye(1),
        b=np.zeros(1),
        C=np.array([[1.0]]),
        d=np.zeros(1),
    )
    config = SolverConfig(
        projection_method="fixed",
        observe_projection_events=True,
    )
    with pytest.raises(ValueError, match="adaptive"):
        SNNSolver(problem, config)

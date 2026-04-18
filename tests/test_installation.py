"""Smoke test: verify ``snn_opt`` is importable and solves a trivial QP.

Runs as a plain script (``python tests/test_installation.py``) and is also
discoverable by ``pytest``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from a clean checkout without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from snn_opt import solve_qp


def test_simple_qp() -> None:
    """min ||x||^2 s.t. x1 + x2 <= 1 — trivial 2D QP, optimal at the origin."""
    A = np.eye(2)
    b = np.zeros(2)
    C = np.array([[1.0, 1.0]])
    d = np.array([-1.0])
    x0 = np.array([1.0, 1.0])

    result = solve_qp(A, b, C, d, x0, k0=0.05, max_iterations=500)

    assert result.final_objective < 1.0, "objective unexpectedly high"
    assert result.constraint_violations[-1] < 1e-5, "constraints violated at solution"


def main() -> int:
    print("Running snn_opt installation smoke test...")
    try:
        test_simple_qp()
    except AssertionError as exc:
        print(f"FAIL: {exc}")
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: {type(exc).__name__}: {exc}")
        import traceback

        traceback.print_exc()
        return 1

    print("PASS — snn_opt is installed and solves the smoke-test QP.")
    print("Try the examples next:")
    print("  python examples/example1_simple_2d.py")
    print("  python examples/example4_warm_start.py")
    print("  python examples/example7_svm_dual.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())

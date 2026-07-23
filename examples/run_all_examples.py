"""Run all example scripts in sequence.

Executes every example and reports a pass/fail summary at the end.

Usage:
    python examples/run_all_examples.py            # run everything
    python examples/run_all_examples.py --pause    # wait for Enter between examples

The runner pauses between examples only when `--pause` is given AND stdin is a
terminal, so it is safe to call from a script, a Makefile or CI. It previously
called `input()` unconditionally, which made it fail with EOFError anywhere
without a TTY.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parent

EXAMPLES = [
    ("example1_simple_2d.py", "Simple 2D quadratic program"),
    ("example1_basic_2d.py", "2D trajectory and spike rug (README figure)"),
    ("example1_advanced_2d.py", "2D diagnostics: spike raster and violations"),
    ("example2_3d_polytope.py", "3D QP with polytope constraints"),
    ("example3_linear_program.py", "Linear program with box constraints"),
    ("example4_warm_start.py", "Warm starting (receding horizon)"),
    ("example5_infeasible_recovery.py", "Infeasible initialization recovery"),
    ("example6_equality_constraint.py", "Equality constraint handling"),
    ("example7_svm_dual.py", "SVM dual (auto k0, bounds as implicit facets)"),
    ("example_raw_mode.py", "Raw vs optimized mode (README figure)"),
]


def main() -> int:
    pause = "--pause" in sys.argv and sys.stdin.isatty()

    print("=" * 78)
    print(f"Running {len(EXAMPLES)} SNN solver examples")
    print("=" * 78)

    failed: list[str] = []
    for i, (script, description) in enumerate(EXAMPLES, 1):
        print(f"\n{'=' * 78}")
        print(f"Example {i}/{len(EXAMPLES)}: {description}")
        print(f"Script: {script}")
        print("=" * 78 + "\n")

        path = EXAMPLES_DIR / script
        if not path.exists():
            print(f"x {script} not found")
            failed.append(script)
            continue

        result = subprocess.run([sys.executable, str(path)], check=False)
        if result.returncode == 0:
            print(f"\nok {script}")
        else:
            print(f"\nx {script} failed with exit code {result.returncode}")
            failed.append(script)

        if pause and i < len(EXAMPLES):
            input("\nPress Enter to continue to the next example...")

    print("\n" + "=" * 78)
    if failed:
        print(f"{len(EXAMPLES) - len(failed)}/{len(EXAMPLES)} passed. Failed:")
        for script in failed:
            print(f"  {script}")
        print("=" * 78)
        return 1
    print(f"All {len(EXAMPLES)} examples completed successfully")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())

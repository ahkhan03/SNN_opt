"""Run every benchmark script and write all figures to ``../figures/``.

Used by ``make figures`` and by CI to regenerate the README/docs imagery.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = sorted(p for p in HERE.glob("[0-9][0-9]_*.py"))


def main() -> int:
    if not SCRIPTS:
        print("no benchmark scripts found", file=sys.stderr)
        return 1
    failures: list[str] = []
    for script in SCRIPTS:
        print(f"\n=== {script.name} ===")
        try:
            runpy.run_path(str(script), run_name="__main__")
        except SystemExit as exc:
            # Each script ends in `sys.exit(main())`. Without catching that,
            # the first script's SystemExit propagates and the runner stops
            # after one figure while still exiting 0, which is how this script
            # silently regenerated only 01_convergence for months.
            if exc.code not in (0, None):
                failures.append(f"{script.name} (exit {exc.code})")
        except Exception as exc:  # noqa: BLE001 - report and keep going
            failures.append(f"{script.name} ({type(exc).__name__}: {exc})")

    if failures:
        print("\nFAILED:", *failures, sep="\n  ", file=sys.stderr)
        return 1
    print(f"\nall {len(SCRIPTS)} benchmark scripts completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

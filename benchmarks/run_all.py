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
    for script in SCRIPTS:
        print(f"\n=== {script.name} ===")
        runpy.run_path(str(script), run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())

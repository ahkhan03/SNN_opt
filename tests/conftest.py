"""Make the repo root importable for tests that use the benchmarks helpers.

`python -m pytest` puts the current directory on sys.path, but a bare
`pytest` invocation (the release check) and the wheels CI test command
(`pytest {project}/tests -q`, run from an arbitrary working directory) do
not, so `from benchmarks.qpref import ...` only resolved by accident of
invocation. Anchor the path to this file instead.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

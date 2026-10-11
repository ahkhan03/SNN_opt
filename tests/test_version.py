"""Version labelling: a checkout reports a release version only at that release's tag.

Results record ``snn_opt.__version__`` as the solver identity, so a development
checkout that reported a released version would label results with code the
release does not contain. ``main`` therefore carries the next ``X.Y.Z.devN``
version, and only the commit a ``vX.Y.Z`` tag points at carries ``X.Y.Z``
(procedure: RELEASING.md).
"""
import re
import subprocess
from pathlib import Path

import pytest

from snn_opt import __version__

ROOT = Path(__file__).resolve().parents[1]


def _git(*args):
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def test_package_and_pyproject_agree():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    declared = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.M).group(1)
    assert declared == __version__


def test_release_version_only_at_its_tag():
    if _git("rev-parse", "--is-inside-work-tree") != "true":
        pytest.skip("not a git checkout (sdist or installed wheel)")
    tag = _git("describe", "--exact-match", "--tags", "--match", "v[0-9]*", "HEAD")
    if tag is not None:
        assert tag == f"v{__version__}", f"HEAD is tagged {tag} but the package says {__version__}"
    else:
        assert ".dev" in __version__, (
            f"untagged commit reports the release version {__version__}: bump to the next "
            f"X.Y.Z.dev0 (see RELEASING.md) or tag this commit as the release")

"""Version identity is derived from git.

An exact ``vX.Y.Z`` tag on a clean tree reports ``X.Y.Z`` with no local
segment. Any other commit, and a dirty tree, reports a ``.devN`` version
whose local node is that commit's SHA; a dirty tree also carries the date
marker. A git checkout never reports the fallback marker, and the
imported version matches a fresh derivation, so a stale editable
``_version.py`` cannot pass. ``CITATION.cff`` is the only hand-edited version
field, and it must equal the tag when HEAD is that tag.
"""

import ast
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

import pytest

from snn_opt import __version__
from snn_opt._version_source import VERSION_NOT_DERIVED

ROOT = Path(__file__).resolve().parents[1]
_EXACT_TAG = re.compile(r"v(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)")


def _git(*args):
    try:
        return subprocess.run(
            ["git", *args],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _in_git_checkout():
    return _git("rev-parse", "--is-inside-work-tree") == "true"


def _citation_version():
    text = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    found = re.search(r'^version:\s*"([^"]+)"', text, re.M)
    assert found, "CITATION.cff has no version"
    return found.group(1)


def _meta_version(text):
    for line in text.splitlines():
        if line.startswith("Version:"):
            return line.split(":", 1)[1].strip()
    raise AssertionError("metadata has no Version field")


def _assigned_version(source):
    tree = ast.parse(source)
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if "version" not in names and "__version__" not in names:
            continue
        value = node.value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            return value.value
    raise AssertionError("no version assignment")


def _toolchain_missing():
    missing = []
    for name in ("build", "wheel", "setuptools", "pybind11", "setuptools_scm"):
        try:
            __import__(name)
        except ImportError:
            missing.append(name)
    return missing


def _extract_sdist(tar, dest):
    # tarfile's data filter arrived in 3.12. Older interpreters still have to
    # run this test inside the wheel matrix.
    if sys.version_info >= (3, 12):
        tar.extractall(dest, filter="data")
    else:
        tar.extractall(dest)


def _plant_source_tree(root, version_py):
    package = root / "src" / "snn_opt"
    package.mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname = 'snn_opt'\n", encoding="utf-8")
    (package / "__init__.py").write_text("", encoding="utf-8")
    source = (ROOT / "src" / "snn_opt" / "_version_source.py").read_text(encoding="utf-8")
    (package / "_version_source.py").write_text(source, encoding="utf-8")
    if version_py is not None:
        (package / "_version.py").write_text(version_py, encoding="utf-8")
    return package


def _resolve_in(src: Path) -> str:
    """Import the planted tree, not the checkout under test."""
    code = (
        "import sys\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "from snn_opt._version_source import resolve_version\n"
        "print(resolve_version())\n"
    )
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("SETUPTOOLS_SCM_PRETEND_VERSION", None)
    proc = subprocess.run(
        [sys.executable, "-c", code, str(src)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip().splitlines()[-1]


def test_fallback_marker_is_not_a_release_number(tmp_path):
    from packaging.version import Version

    assert VERSION_NOT_DERIVED != "unknown"
    assert re.fullmatch(r"\d+\.\d+\.\d+", VERSION_NOT_DERIVED) is None
    parsed = Version(VERSION_NOT_DERIVED)
    assert parsed.public == "0.0.0"
    assert parsed.local is not None

    bare = tmp_path / "bare"
    _plant_source_tree(bare, version_py=None)
    assert not (bare / ".git").exists()
    assert not (bare / "src" / "snn_opt" / "_version.py").exists()
    assert _resolve_in(bare / "src") == VERSION_NOT_DERIVED

    broken = tmp_path / "broken"
    _plant_source_tree(broken, version_py="version = (\n")
    assert _resolve_in(broken / "src") == VERSION_NOT_DERIVED


def test_packaged_version_beats_the_marker_without_git(tmp_path):
    root = tmp_path / "shipped"
    _plant_source_tree(root, version_py="version = '4.5.6'\n")
    assert not (root / ".git").exists()
    assert _resolve_in(root / "src") == "4.5.6"


def test_symlinked_package_dir_uses_the_checkout(tmp_path):
    alt = tmp_path / "alt" / "snn_opt"
    alt.mkdir(parents=True)
    (alt / "__init__.py").write_text("", encoding="utf-8")
    source = (ROOT / "src" / "snn_opt" / "_version_source.py").read_text(encoding="utf-8")
    (alt / "_version_source.py").write_text(source, encoding="utf-8")
    (alt / "_version.py").write_text("version = '9.9.9'\n", encoding="utf-8")

    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    link = repo / "src" / "snn_opt"
    try:
        link.symlink_to(alt, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink layout is not available: {exc}")
    (repo / "pyproject.toml").write_text("[project]\nname = 'snn_opt'\n", encoding="utf-8")
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"],
        cwd=repo,
        check=True,
    )
    subprocess.run(["git", "config", "user.name", "test"], cwd=repo, check=True)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(
        ["git", "commit", "-m", "scratch symlink checkout"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(["git", "tag", "v1.2.3"], cwd=repo, check=True)
    assert _resolve_in(repo / "src") == "1.2.3"


def test_version_matches_git():
    if not _in_git_checkout():
        pytest.skip("not a git checkout (sdist or installed wheel)")
    from setuptools_scm import get_version

    derived = get_version(root=ROOT)
    assert __version__ == derived
    assert VERSION_NOT_DERIVED not in __version__
    assert __version__ != "unknown"

    exact = _git("describe", "--exact-match", "--tags", "--match", "v[0-9]*", "HEAD")
    sha = _git("rev-parse", "HEAD")
    assert sha
    matched = _EXACT_TAG.fullmatch(exact) if exact else None
    # Untracked files do not dirty setuptools_scm. A dirty tracked file on
    # the tagged commit does: guess-next-dev then reports the next patch
    # as .dev0 plus the node and the date marker, not the bare release.
    dirty = _git("status", "--porcelain", "--untracked-files=no")
    if matched and not dirty:
        release = f"{matched.group(1)}.{matched.group(2)}.{matched.group(3)}"
        assert _citation_version() == release
        assert __version__ == release
        assert "+" not in __version__
        assert ".dev" not in __version__
    else:
        if matched:
            release = f"{matched.group(1)}.{matched.group(2)}.{matched.group(3)}"
            assert _citation_version() == release
            assert __version__ != release
            assert re.search(r"\.d\d{8}$", __version__), __version__
        assert ".dev" in __version__, __version__
        node = re.search(r"\+g([0-9a-fA-F]+)", __version__)
        assert node, __version__
        assert sha.startswith(node.group(1))


def _outside_any_git_repo(path: Path):
    return not any((parent / ".git").exists() for parent in [path, *path.parents])


def test_sdist_rebuilds_to_the_same_version_without_git():
    """An sdist carries its version, so a rebuild with no git reproduces it."""
    missing = _toolchain_missing()
    if missing:
        pytest.skip("build toolchain not importable: " + ", ".join(missing))
    if not _in_git_checkout():
        pytest.skip("need a git checkout to build the source distribution")
    from setuptools_scm import get_version

    derived = get_version(root=ROOT)
    env = os.environ.copy()
    env.pop("SETUPTOOLS_SCM_PRETEND_VERSION", None)
    with tempfile.TemporaryDirectory(prefix="snn-opt-sdist-") as raw:
        tmp = Path(raw)
        if not _outside_any_git_repo(tmp):
            pytest.skip("default temporary directory is inside a git worktree")
        sdist_dir = tmp / "sdist"
        sdist_dir.mkdir()
        subprocess.run(
            [
                sys.executable,
                "-m",
                "build",
                "--sdist",
                "--no-isolation",
                "--outdir",
                str(sdist_dir),
            ],
            cwd=ROOT,
            check=True,
            env=env,
        )
        sdists = list(sdist_dir.glob("snn_opt-*.tar.gz"))
        assert len(sdists) == 1, sdists
        with tarfile.open(sdists[0]) as tar:
            infos = [
                m.name
                for m in tar.getmembers()
                if m.name.endswith("/PKG-INFO") and m.name.count("/") == 1
            ]
            assert len(infos) == 1, infos
            sdist_version = _meta_version(tar.extractfile(infos[0]).read().decode())
            packaged = [
                m.name for m in tar.getmembers() if m.name.endswith("src/snn_opt/_version.py")
            ]
            assert len(packaged) == 1, packaged
            assert _assigned_version(tar.extractfile(packaged[0]).read().decode()) == sdist_version
            names = [m.name for m in tar.getmembers()]
            tops = {parts[1] for name in names if len(parts := name.split("/")) > 1}
            assert "tests" not in tops, names
            assert "tools" not in tops, names
            assert any(name.endswith("src/snn_opt/__init__.py") for name in names)
            assert any(name.endswith("src/snn_opt/_native/bindings.cpp") for name in names)
            extract = tmp / "src"
            extract.mkdir()
            _extract_sdist(tar, extract)
        roots = [p for p in extract.iterdir() if p.is_dir()]
        assert len(roots) == 1, roots
        assert not (roots[0] / ".git").exists()
        assert _outside_any_git_repo(roots[0])
        wheel_dir = tmp / "wheel"
        wheel_dir.mkdir()
        subprocess.run(
            [
                sys.executable,
                "-m",
                "build",
                "--wheel",
                "--no-isolation",
                "--outdir",
                str(wheel_dir),
            ],
            cwd=roots[0],
            check=True,
            env=env,
        )
        wheels = list(wheel_dir.glob("snn_opt-*.whl"))
        assert len(wheels) == 1, wheels
        with zipfile.ZipFile(wheels[0]) as zf:
            meta = [n for n in zf.namelist() if n.endswith(".dist-info/METADATA")]
            vers = [n for n in zf.namelist() if n.endswith("snn_opt/_version.py")]
            assert len(meta) == 1 and len(vers) == 1
            wheel_version = _meta_version(zf.read(meta[0]).decode())
            assert _assigned_version(zf.read(vers[0]).decode()) == wheel_version
        assert sdist_version == derived
        assert wheel_version == derived

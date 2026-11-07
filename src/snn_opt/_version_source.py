"""Resolve ``__version__`` for a checkout or an installed artifact.

Live derivation wins when this module is loaded from a git checkout and
``setuptools_scm`` imports. ``get_version()`` then names the commit, so a
stale ``_version.py`` cannot hide it. Anything else uses the packaged
``_version.py`` written at the last build. A source tree without ``.git``
has no commit to name, and that file describes the code it was shipped
with, so it beats the marker. An editable install reports the version as
of its last build: ``setuptools_scm`` stays in the build environment and
is not required at runtime. The marker is only what remains when neither
a derivable checkout nor a usable packaged file is available.
"""

import sys
from pathlib import Path

# A PEP 440 local version. It is not a release number and not the word
# "unknown". ``reference.py`` copies ``__version__`` into every certified record.
VERSION_NOT_DERIVED = "0.0.0+invalid.snn_opt.version.not.derived"


def _checkout_root(src_dir: Path, package_dir: Path):
    """Return the project root when ``src_dir`` is a ``src`` layout of this package."""
    if src_dir.name != "src":
        return None
    root = src_dir.parent
    try:
        if (src_dir / "snn_opt").resolve() != package_dir:
            return None
    except OSError:
        return None
    if not (root / "pyproject.toml").is_file():
        return None
    if not (root / "src" / "snn_opt" / "__init__.py").is_file():
        return None
    return root


def _source_root():
    """Project root when this module is the checkout under ``src/``, else None.

    An installed copy lives in ``site-packages`` and must not walk into a
    parent checkout: a venv inside the repo would otherwise report the live
    tree instead of the wheel that was installed.

    ``Path.resolve()`` follows a symlinked package directory, so
    ``repo/src/snn_opt -> /alt/snn_opt`` does not look like a ``src`` layout
    from ``__file__`` alone. A ``sys.path`` entry that has this package as
    ``snn_opt``, is itself a ``src`` directory, and has ``pyproject.toml``
    in its parent, is still that checkout.
    """
    package_dir = Path(__file__).resolve().parent
    found = _checkout_root(package_dir.parent, package_dir)
    if found is not None:
        return found
    for entry in sys.path:
        if not entry:
            continue
        found = _checkout_root(Path(entry), package_dir)
        if found is not None:
            return found
    return None


def _packaged_version():
    try:
        from ._version import version as packaged
    except Exception:
        return None
    if not isinstance(packaged, str):
        return None
    text = packaged.strip()
    if not text or text.lower() == "unknown" or text == VERSION_NOT_DERIVED:
        return None
    return text


def resolve_version():
    root = _source_root()
    if root is not None and (root / ".git").exists():
        try:
            from setuptools_scm import get_version
        except ImportError:
            derived = None
        else:
            try:
                # Defaults are guess-next-dev and node-and-date, the same
                # defaults the build uses. Do not pass version_file: importing
                # must not rewrite the tree.
                derived = get_version(root=root)
            except Exception:
                derived = None
        if (
            isinstance(derived, str)
            and derived.strip()
            and derived.strip() != VERSION_NOT_DERIVED
            and derived.strip().lower() != "unknown"
        ):
            return derived.strip()
    # No live derivation. The packaged file is the version of the code it was
    # shipped with, including an editable install's last build and an extracted
    # sdist. The marker is last.
    packaged = _packaged_version()
    if packaged is None:
        return VERSION_NOT_DERIVED
    return packaged

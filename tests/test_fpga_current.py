"""fpga/CURRENT names only paths that exist, so the resolver cannot rot silently."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ("polyhedral", "conic")
PATH_FIELDS = ("package", "boundary", "cpu_baseline", "datapath_width_source")


class CurrentParseError(ValueError):
    """fpga/CURRENT repeated a block name or a field. Raised, not asserted,
    so the guard still runs under python -O."""


def _parse(text: str) -> dict[str, dict[str, str]]:
    blocks: dict[str, dict[str, str]] = {}
    current = None
    current_name = None
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, _, value = line.strip().partition(":")
        if not line.startswith(" "):
            if value.strip():
                raise CurrentParseError(
                    f"top-level key {key!r} must open a block")
            if key in blocks:
                raise CurrentParseError(f"repeated block name {key!r}")
            current_name = key
            current = {}
            blocks[key] = current
        else:
            if current is None:
                raise CurrentParseError(f"field {key!r} outside a block")
            if key in current:
                raise CurrentParseError(
                    f"repeated field {key!r} in block {current_name!r}")
            current[key] = value.strip()
    return blocks


def test_current_names_existing_paths_per_problem_class():
    blocks = _parse((ROOT / "fpga" / "CURRENT").read_text())
    assert set(blocks) == set(CLASSES)
    for name in CLASSES:
        fields = blocks[name]
        assert set(PATH_FIELDS) <= set(fields), name
        package = fields["package"]
        assert (ROOT / package).is_dir(), f"{name}: {package}"
        for field in PATH_FIELDS[1:]:
            value = fields[field]
            if field == "cpu_baseline" and value == "none":
                continue
            assert (ROOT / value).is_file(), f"{name}.{field}: {value}"
        assert fields["boundary"].startswith(package + "/")
        assert fields["datapath_width_source"].startswith(package + "/")


def test_lineage_packages_resolve_like_the_default():
    """A lineage-prefixed field (bram_package, ...) is a full package entry.

    The polyhedral class names more than one qualified package: FULL at
    n,m <= 128 uses the BRAM lineage, and the default package covers a
    larger problem or a non-FULL route. Every <prefix>_package carries its
    own <prefix>_boundary and <prefix>_datapath_width_source inside that
    package, checked like the default fields; a lineage that names a path
    that does not exist fails here rather than at paper-snapshot time.
    """
    blocks = _parse((ROOT / "fpga" / "CURRENT").read_text())
    for name in CLASSES:
        fields = blocks[name]
        prefixes = [k[: -len("_package")] for k in fields
                    if k.endswith("_package") and k != "package"]
        for prefix in prefixes:
            package = fields[f"{prefix}_package"]
            assert (ROOT / package).is_dir(), f"{name}.{prefix}_package: {package}"
            for suffix in ("boundary", "datapath_width_source"):
                field = f"{prefix}_{suffix}"
                assert field in fields, f"{name}.{field} missing"
                value = fields[field]
                assert (ROOT / value).is_file(), f"{name}.{field}: {value}"
                assert value.startswith(package + "/"), f"{name}.{field}"


def test_repeated_block_name_fails():
    with pytest.raises(
            CurrentParseError, match="repeated block name 'polyhedral'"):
        _parse("polyhedral:\n  package: a\npolyhedral:\n  package: b\n")


def test_repeated_field_fails():
    with pytest.raises(
            CurrentParseError, match="repeated field 'bram_package'"):
        _parse(
            "polyhedral:\n"
            "  package: a\n"
            "  bram_package: b\n"
            "  bram_package: c\n"
        )


def test_shipped_duplicate_bram_package_fails():
    """64dae00:fpga/CURRENT repeats bram_package inside polyhedral."""
    text = subprocess.check_output(
        ["git", "show", "64dae00:fpga/CURRENT"],
        cwd=ROOT,
        text=True,
    )
    with pytest.raises(CurrentParseError) as exc_info:
        _parse(text)
    assert str(exc_info.value) == (
        "repeated field 'bram_package' in block 'polyhedral'")

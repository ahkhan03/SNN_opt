"""fpga/CURRENT names only paths that exist, so the resolver cannot rot silently."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ("polyhedral", "conic")
PATH_FIELDS = ("package", "boundary", "cpu_baseline", "datapath_width_source")


def _parse(text: str) -> dict[str, dict[str, str]]:
    blocks: dict[str, dict[str, str]] = {}
    current = None
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, _, value = line.strip().partition(":")
        if not line.startswith(" "):
            assert not value.strip(), f"top-level key {key!r} must open a block"
            current = blocks.setdefault(key, {})
        else:
            assert current is not None, f"field {key!r} outside a block"
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

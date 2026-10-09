"""The generated capability matrix: declared expectations, freshness, FPGA citations."""

import importlib.util
from pathlib import Path

import pytest

from snn_opt import capabilities as cap

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("render_capabilities",
                                               ROOT / "tools" / "render_capabilities.py")
render_capabilities = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(render_capabilities)

CELLS = [(row, j) for row in cap.ROWS for j in range(len(cap.COLUMNS))
         if row.expected(j) != "n/a"]


@pytest.mark.parametrize("row,column", CELLS,
                         ids=[f"{r.set_type}-{'-'.join(cap.COLUMNS[j])}" for r, j in CELLS])
def test_cell_matches_declared_expectation(row, column):
    backend = cap.COLUMNS[column][1]
    if not cap.backend_available(backend):
        pytest.skip(f"this build cannot run backend={backend!r}")
    outcome = cap.probe(row, column)
    assert outcome.status == row.expected(column), outcome.detail


def test_rendered_matrix_is_fresh():
    if not all(cap.backend_available(b) for _, b in cap.COLUMNS):
        pytest.skip("rendering needs the compiled kernel with OpenMP")
    text, mismatches = render_capabilities.render()
    assert not mismatches, mismatches
    committed = (ROOT / "docs" / "capabilities.md").read_text()
    assert text == committed, "docs/capabilities.md is stale: run tools/render_capabilities.py"


def test_fpga_boundaries_are_declared_and_cited():
    for package, entry in cap.FPGA_BOUNDARIES.items():
        assert (ROOT / package).is_dir(), package
        for path in entry["evidence"]:
            assert (ROOT / path).exists(), f"{package} cites missing {path}"
        assert set(entry) == {"evidence", *cap.FPGA_FEATURES}, package
        for feature in cap.FPGA_FEATURES:
            assert entry[feature] in cap.FPGA_VALUES, (package, feature, entry[feature])
    for package in render_capabilities.current_packages():
        assert package.rstrip("/") in cap.FPGA_BOUNDARIES, (
            f"fpga/CURRENT names {package} but FPGA_BOUNDARIES does not declare it")


def test_row_expectations_are_well_formed():
    for row in cap.ROWS:
        codes = row.expect.split()
        assert len(codes) == len(cap.COLUMNS), row.set_type
        assert set(codes) <= set(cap.CODES), row.set_type

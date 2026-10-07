"""Regression tests for the exact scaled-SOC projector."""

import numpy as np

from snn_opt import scaled_soc_projector


def test_scaled_soc_inside_and_radial_cases():
    cand = scaled_soc_projector(1, [2, 3], 0.5)
    inside = np.array([7.0, 2.0, 0.5, 0.2, -3.0])
    np.testing.assert_array_equal(cand.project(inside), inside)
    x = np.array([7.0, 1.0, 3.0, 4.0, -3.0])
    y = cand.project(x)
    assert np.linalg.norm(y[[2, 3]]) <= 0.5 * y[1] + 1e-12
    np.testing.assert_array_equal(y[[0, 4]], x[[0, 4]])
    expected_t = (x[1] + 0.5 * np.linalg.norm(x[[2, 3]])) / 1.25
    np.testing.assert_allclose(y[1], expected_t)


def test_scaled_soc_axial_and_zero_cases_and_normal_metadata():
    cand = scaled_soc_projector(0, [1, 2], 0.8)
    axial = np.array([-2.0, 1.0, -1.0])
    np.testing.assert_array_equal(cand.project(axial), np.zeros(3))
    zero = np.zeros(3)
    np.testing.assert_array_equal(cand.project(zero), zero)
    boundary = np.array([1.0, 0.8, 0.0])
    normal = cand.normal(boundary)
    assert normal is not None
    np.testing.assert_allclose(normal, np.array([-0.8, 1.0, 0.0]) / np.sqrt(1.64))
    assert cand.kkt_data["set"] == "scaled_soc"


def test_scaled_soc_unequal_slope_negative_scalar_projects_to_apex():
    cand = scaled_soc_projector(0, [1, 2], 0.2)
    # Here ||z|| is greater than -mu*t, but the boundary-ray parameter is
    # negative.  The exact Euclidean projection is still the apex.
    x = np.array([-0.13210486, 0.64042265, 0.10490012])
    np.testing.assert_allclose(cand.project(x), np.zeros(3), atol=1e-12)

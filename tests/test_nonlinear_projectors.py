"""Closed-form ball, SOC and box projector fixtures."""

import numpy as np
import pytest

from snn_opt import (
    AffineSubspaceProjector,
    DykstraProjector,
    ball_projector,
    box_projector,
    halfspace_projector,
    soc_projector,
)


def test_ball_soc_subset_projectors_match_direct_projection():
    x = np.array([9.0, 3.0, 4.0, -2.0, 0.5])
    ball = ball_projector([1, 3], radius=2.0, center=np.array([1.0, -1.0]))
    y = ball.project(x)
    delta = x[[1, 3]] - np.array([1.0, -1.0])
    expected_local = np.array([1.0, -1.0]) + 2.0 * delta / np.linalg.norm(delta)
    np.testing.assert_allclose(y[[1, 3]], expected_local)
    np.testing.assert_array_equal(y[[0, 2, 4]], x[[0, 2, 4]])

    soc = soc_projector(0, [2, 4])
    soc_x = x.copy()
    soc_x[0] = 0.5
    z = soc.project(soc_x)
    t, v = soc_x[0], soc_x[[2, 4]]
    vn = np.linalg.norm(v)
    alpha = (vn + t) / 2.0
    expected_t = alpha
    expected_v = alpha * v / vn
    np.testing.assert_allclose(z[0], expected_t)
    np.testing.assert_allclose(z[[2, 4]], expected_v)
    np.testing.assert_array_equal(z[[1, 3]], soc_x[[1, 3]])

    # The affine-subspace helper uses the standard minimum-distance formula.
    sub = AffineSubspaceProjector(np.array([[1.0, 1.0]]), np.array([1.0]))
    q = sub.project(np.array([0.0, 0.0]))
    np.testing.assert_allclose(q, [0.5, 0.5])
    np.testing.assert_allclose(sub.B @ q, [1.0])


def test_box_dykstra_hand_case_matches_closed_form():
    # [0,1]^2 intersected with x0 + x1 <= 0.5: the projections are hand-checkable.
    dykstra = DykstraProjector((box_projector(0.0, 1.0), halfspace_projector([1.0, 1.0], -0.5)))
    np.testing.assert_allclose(dykstra.project(np.array([2.0, 2.0])), [0.25, 0.25], atol=1e-10)
    np.testing.assert_allclose(dykstra.project(np.array([1.5, -3.0])), [0.5, 0.0], atol=1e-10)
    assert dykstra.last_diagnostics["converged"]


def test_box_projector_clip_slack_normal_and_scoping():
    box = box_projector([0.0, -1.0], [1.0, 2.0], coordinates=[1, 3])
    x = np.array([9.0, 3.0, 4.0, -2.0, 0.5])
    y = box.project(x)
    np.testing.assert_array_equal(y, [9.0, 1.0, 4.0, -1.0, 0.5])
    np.testing.assert_array_equal(x, [9.0, 3.0, 4.0, -2.0, 0.5])
    assert box.kkt_data["set"] == "box"
    assert box.coordinates == (1, 3)
    # Coordinate 1 is 2 above its upper bound: slack -2, outward normal +e1.
    assert box.kkt_data["slack"](x) == pytest.approx(-2.0)
    np.testing.assert_array_equal(box.normal(x), [0.0, 1.0, 0.0, 0.0, 0.0])
    # Coordinate 3 below its lower bound is the tightest: outward normal -e3.
    x2 = np.array([0.0, 0.5, 0.0, -4.0, 0.0])
    np.testing.assert_array_equal(box.normal(x2), [0.0, 0.0, 0.0, -1.0, 0.0])
    assert box.normal(np.array([0.0, 0.5, 0.0, 0.5, 0.0])) is None
    assert box.kkt_data["slack"](np.array([0.0, 0.5, 0.0, 0.5, 0.0])) == pytest.approx(0.5)


def test_box_projector_one_sided_and_ambient_bounds():
    lower_only = box_projector(0.0, np.inf)
    np.testing.assert_array_equal(lower_only.project(np.array([-1.0, 5.0, 1e300])),
                                  [0.0, 5.0, 1e300])
    upper_only = box_projector(-np.inf, [1.0, 2.0])
    np.testing.assert_array_equal(upper_only.project(np.array([3.0, -7.0])), [1.0, -7.0])
    with pytest.raises(ValueError, match="dimension 2"):
        upper_only.project(np.zeros(3))
    with pytest.raises(ValueError, match="exceeds state dimension"):
        box_projector(0.0, 1.0, coordinates=[4]).project(np.zeros(2))


@pytest.mark.parametrize(
    "lower,upper,coordinates,match",
    [
        (1.0, 0.0, None, "exceeds upper"),
        ([0.0, 2.0], [1.0, 1.0], None, "exceeds upper"),
        ([0.0, 0.0, 0.0], [1.0, 1.0], None, "different lengths"),
        ([0.0, 0.0, 0.0], 1.0, [0, 1], "one entry per coordinate"),
        (np.nan, 1.0, None, "NaN"),
        (0.0, np.nan, None, "NaN"),
        (np.inf, np.inf, None, "non-empty"),
        (-np.inf, -np.inf, None, "non-empty"),
        ([], 1.0, None, "empty"),
    ],
)
def test_box_projector_rejects_bad_bounds(lower, upper, coordinates, match):
    with pytest.raises(ValueError, match=match):
        box_projector(lower, upper, coordinates=coordinates)

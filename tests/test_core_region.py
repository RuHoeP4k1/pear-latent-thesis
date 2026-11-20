import numpy as np
import pytest

from pearlatent.core_region import (
    apply_fill,
    body_height,
    core_region,
    core_regions,
    fit_fill,
)


def box_body(first: int, last: int, n: int = 32) -> np.ndarray:
    """A square column of body cells from slice `first` to `last` (inclusive)."""
    body = np.zeros((n, n, n), dtype=bool)
    body[10:20, 10:20, first : last + 1] = True
    return body


def test_height_runs_over_the_body_not_the_grid():
    # Body on slices 2 to 21: 20 slices, so slice 2 has height 0 and slice 12 height 0.5.
    height = body_height(box_body(2, 21))
    assert height[15, 15, 2] == 0.0
    assert height[15, 15, 12] == pytest.approx(0.5)
    assert height[15, 15, 21] == pytest.approx(19 / 20)
    assert np.isnan(height[0, 0, 12])  # outside the body
    assert np.isnan(height[15, 15, 25])  # above the body (where a stalk would be)


def test_region_keeps_the_slices_between_the_bounds_at_all_radii():
    body = box_body(0, 19)  # 20 slices: slice k has height k / 20
    region = core_region(body)
    # 0.05 <= k / 20 < 0.60  ->  slices 1 to 11.
    kept_slices = np.flatnonzero(region.any(axis=(0, 1)))
    np.testing.assert_array_equal(kept_slices, np.arange(1, 12))
    # Every body cell of a kept slice is kept, from the axis to the edge.
    np.testing.assert_array_equal(region[:, :, 5], body[:, :, 5])
    assert not (region & ~body).any()


def test_regions_follow_each_fruit_body():
    bodies = np.stack([box_body(0, 19), box_body(0, 29)])
    regions = core_regions(bodies)
    assert regions.shape == bodies.shape
    # The longer body reaches higher slices: 0.60 * 30 = 18, so slices up to 17.
    assert np.flatnonzero(regions[1].any(axis=(0, 1))).max() == 17
    assert np.flatnonzero(regions[0].any(axis=(0, 1))).max() == 11


def test_empty_body_is_refused():
    with pytest.raises(ValueError):
        body_height(np.zeros((32, 32, 32), dtype=bool))


def test_fill_is_the_mean_over_fruit_with_the_cell_inside_their_region():
    x = np.array([[1.0, 10.0, 5.0], [3.0, 20.0, 6.0], [8.0, 30.0, 7.0]])
    regions = np.array(
        [[True, True, False], [True, False, False], [False, False, False]]
    )
    fill = fit_fill(x, regions)
    # Cell 0: fruit 0 and 1 -> 2. Cell 1: fruit 0 only -> 10. Cell 2: none -> 0.
    np.testing.assert_allclose(fill, [2.0, 10.0, 0.0])


def test_cells_outside_the_region_are_zero_after_centring():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 50))
    regions = rng.random((40, 50)) < 0.6
    filled = apply_fill(x, regions, fit_fill(x, regions))
    centred = filled - filled.mean(axis=0)
    np.testing.assert_allclose(centred[~regions], 0.0, atol=1e-12)
    # Cells inside the region keep their own values.
    np.testing.assert_array_equal(filled[regions], x[regions])


def test_held_out_fruit_is_filled_from_training_values_only():
    x_train = np.array([[1.0, 2.0], [3.0, 4.0]])
    regions_train = np.ones((2, 2), dtype=bool)
    x_held_out = np.array([[100.0, 200.0]])
    fill = fit_fill(x_train, regions_train)
    filled = apply_fill(x_held_out, np.array([[True, False]]), fill)
    # Cell 0 is inside its region and keeps 100; cell 1 gets the training mean 3, not 200.
    np.testing.assert_allclose(filled, [[100.0, 3.0]])
    assert x_held_out[0, 1] == 200.0  # input not changed


def test_regions_may_be_given_as_grids():
    bodies = np.stack([box_body(0, 19), box_body(0, 29)])
    regions = core_regions(bodies)
    x = np.arange(2 * 32**3, dtype=float).reshape(2, -1)
    fill = fit_fill(x, regions)
    assert fill.shape == (32**3,)
    filled = apply_fill(x, regions, fill)
    np.testing.assert_array_equal(
        filled[regions.reshape(2, -1)], x[regions.reshape(2, -1)]
    )

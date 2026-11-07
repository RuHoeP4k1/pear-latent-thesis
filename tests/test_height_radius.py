import numpy as np
import pytest

from pearlatent.height_radius import fill_empty_bins, height_radius_features


def test_empty_bin_takes_the_mean_of_its_nearest_neighbours():
    means = np.array([[1.0, np.nan, 3.0], [5.0, 6.0, 7.0]])
    filled, n = fill_empty_bins(means)
    assert n == 1
    # (0, 1) has three neighbours at distance 1: 1, 3 and 6.
    assert filled[0, 1] == pytest.approx((1 + 3 + 6) / 3)
    np.testing.assert_array_equal(filled[1], means[1])


def test_all_empty_map_is_refused():
    with pytest.raises(ValueError):
        fill_empty_bins(np.full((2, 2), np.nan))


def test_features_do_not_depend_on_rotation_about_the_axis():
    """A cylinder whose value rises with radius: rotating it 90 degrees gives the same bins."""
    n = 32
    x, y, z = np.meshgrid(*(np.arange(n) for _ in range(3)), indexing="ij")
    radius = np.hypot(x - 15.5, y - 15.5)
    body = (radius < 10) & (z >= 4) & (z < 28)
    grid = np.where(body, radius + 0.1 * x, 0.0)  # an angle-dependent part as well
    rotated_grid = np.rot90(grid, axes=(0, 1))
    rotated_body = np.rot90(body, axes=(0, 1))
    features, n_filled = height_radius_features(
        np.stack([grid, rotated_grid]), np.stack([body, rotated_body])
    )
    assert features.shape == (2, 160)
    assert not np.isnan(features).any()
    np.testing.assert_allclose(features[0], features[1], atol=1e-5)
    # Value rises from the axis to the skin in every height bin.
    profile = features[0].reshape(20, 8)
    assert (np.diff(profile, axis=1) > 0).all()

import numpy as np
import pytest

from pearlatent.pooling import block_mean
from pearlatent.reconstruction import central_planes
from pearlatent.step0 import (
    N_HEIGHT_BINS,
    body_slice_range,
    contrast_retention,
    plane_measures,
)


def _cylinder(core_value: float) -> np.ndarray:
    """Cylinder along the third axis (radius 24, slices 20 to 99), 0.8 outside the core and
    `core_value` within 6 voxels of the axis."""
    gx, gy = np.meshgrid(np.arange(128), np.arange(128), indexing="ij")
    distance = np.sqrt((gx - 64) ** 2 + (gy - 64) ** 2)
    slice_ = np.where(distance <= 24, 0.8, 0.0)
    slice_ = np.where(distance <= 6, core_value, slice_)
    vol = np.zeros((128, 128, 128), dtype=np.float32)
    vol[:, :, 20:100] = slice_[:, :, None]
    return vol


def _geometry():
    return {
        "plane_x": 64,
        "plane_y": 64,
        "axis_x": 64.0,
        "axis_y": 64.0,
        "core_radius_voxels": 8.0,
    }


def _measure(ct, rec):
    cells = block_mean((ct > 0.1).astype(np.float32)) >= 0.5
    planes = np.stack(
        [central_planes(ct, (64, 64, 64)), central_planes(rec, (64, 64, 64))]
    )
    return plane_measures(planes, cells, _geometry())


def test_body_slice_range():
    cells = np.zeros((32, 32, 32), dtype=bool)
    cells[10, 10, 5:25] = True
    assert body_slice_range(cells) == (20, 99)


def test_core_means_and_map_sums():
    ct = _cylinder(core_value=0.5)
    rec = _cylinder(core_value=0.65)
    numbers, sums = _measure(ct, rec)
    # Core radius 8: voxels within 6 of the axis are 0.5 (CT), the ring from 6 to 8 is 0.8.
    assert 0.5 < numbers["core_ct"] < 0.8
    assert numbers["core_reconstruction"] > numbers["core_ct"]
    counts, ct_sum, rec_sum, residual_sum = sums
    assert counts.sum() > 0
    # The residual sums are reconstruction minus CT, bin by bin.
    np.testing.assert_allclose(residual_sum, rec_sum - ct_sum, atol=1e-6)
    # The whole height of the body is covered.
    assert (counts.sum(axis=1) > 0).sum() == N_HEIGHT_BINS
    # Positive residual (reconstruction brighter) only near the axis (radius bins 0 to 4 of
    # 24 cover radius 0 to 0.25 of the outermost body voxel, 6 / 24 = 0.25).
    mean_residual = residual_sum.sum(axis=0) / np.maximum(counts.sum(axis=0), 1)
    assert mean_residual[:5].max() > 0.1
    assert np.abs(mean_residual[8:]).max() < 1e-6


def test_contrast_retention_half():
    healthy = _measure(_cylinder(0.8), _cylinder(0.8))[0]
    damaged = _measure(_cylinder(0.5), _cylinder(0.65))[0]
    core_ct = np.array([healthy["core_ct"], damaged["core_ct"]])
    core_rec = np.array(
        [healthy["core_reconstruction"], damaged["core_reconstruction"]]
    )
    result = contrast_retention(core_ct, core_rec, np.array([0, 3]), 3)
    assert result["ct_difference"] < 0
    # The reconstruction keeps half of the darkening (0.15 of 0.3 in the inner core).
    assert result["retention"] == pytest.approx(0.5, abs=0.01)


@pytest.mark.parametrize("radius_method", ["fruit", "slice"])
def test_latent_map_sums_places_cells_by_height_and_radius(radius_method):
    from pearlatent.step0 import (
        N_LATENT_HEIGHT_BINS,
        N_LATENT_RADIUS_BINS,
        latent_map_sums,
    )

    # Body: cylinder of radius 6 cells around (16, 16), slices 4 to 27 (24 slices).
    gx, gy = np.meshgrid(np.arange(32), np.arange(32), indexing="ij")
    distance = np.sqrt((gx - 16) ** 2 + (gy - 16) ** 2)
    body = np.zeros((32, 32, 32), dtype=bool)
    body[:, :, 4:28] = (distance <= 6)[:, :, None]
    # Latent: 5 on the axis column, 1 elsewhere in the body, -100 outside (must be ignored).
    latent = np.where(body, 1.0, -100.0)
    latent[16, 16, :] = np.where(body[16, 16, :], 5.0, -100.0)

    # On a straight cylinder both radius methods give the same coordinates.
    counts, sums = latent_map_sums(latent, body, radius_method)
    assert counts.shape == (N_LATENT_HEIGHT_BINS, N_LATENT_RADIUS_BINS)
    # Every body cell is counted once (radius 0 to 1, the outermost cell included).
    assert counts.sum() == body.sum()
    # Every height bin is covered.
    assert (counts.sum(axis=1) > 0).all()
    mean = sums / np.maximum(counts, 1)
    # The axis column lands in radius bin 0, mixed with the cells closest to the axis.
    assert mean[:, 0].min() > 1.0
    # Away from the axis only the value 1 remains; the outside value never enters.
    covered = counts[:, 3:] > 0
    np.testing.assert_allclose(mean[:, 3:][covered], 1.0)


def _cone_body() -> np.ndarray:
    """Body cells of a cone along the third axis: radius 10 cells at slice 2, falling by 0.3
    cell per slice to 1.3 at slice 31; the centre of each slice moves from x = 14 to x = 18."""
    gx, gy = np.meshgrid(np.arange(32), np.arange(32), indexing="ij")
    body = np.zeros((32, 32, 32), dtype=bool)
    for k in range(2, 32):
        centre_x = 14 + 4 * (k - 2) / 29
        body[:, :, k] = (gx - centre_x) ** 2 + (gy - 16) ** 2 <= (
            10 - 0.3 * (k - 2)
        ) ** 2
    return body


def test_latent_map_sums_slice_method_puts_the_skin_near_radius_one():
    from pearlatent.step0 import MAX_RADIUS, latent_map_sums

    body = _cone_body()
    # Latent = 1 in the outer ring of each slice (the two outermost cells), 0 inside.
    from scipy import ndimage

    eroded = np.stack(
        [ndimage.binary_erosion(body[:, :, k], iterations=2) for k in range(32)], axis=2
    )
    latent = np.where(body & ~eroded, 1.0, 0.0)

    n_radius = 8
    bin_width = MAX_RADIUS / n_radius
    slice_counts, slice_sums = latent_map_sums(latent, body, "slice", 10, n_radius)
    fruit_counts, _ = latent_map_sums(latent, body, "fruit", 10, n_radius)
    # Radius never exceeds 1, so every body cell is binned, in both methods.
    assert slice_counts.sum() == fruit_counts.sum() == body.sum()

    # Slice method: in every height bin the outermost occupied radius bin is near 1, even at
    # the narrow top; the skin ring dominates there.
    for h in range(10):
        outer = np.flatnonzero(slice_counts[h])[-1]
        assert 0.85 <= outer * bin_width <= MAX_RADIUS
    # Fruit method: the narrow top only reaches small radii (shape is kept).
    top_outer = np.flatnonzero(fruit_counts[-1])[-1]
    assert top_outer * bin_width < 0.5
    # Slice method: the inner bins of the wide part hold no skin cells, despite the moving
    # centre (one straight axis would put skin cells of shifted slices near the axis).
    mean = slice_sums / np.maximum(slice_counts, 1)
    assert mean[:5, 0].max() == 0.0


def test_latent_map_sums_rejects_an_unknown_radius_method():
    from pearlatent.step0 import latent_map_sums

    with pytest.raises(ValueError):
        latent_map_sums(np.zeros((32, 32, 32)), _cone_body(), "angle")


def test_fruit_bin_means_broadcasts_counts_over_value_axes():
    from pearlatent.step0 import fruit_bin_means

    counts = np.array([[[2.0, 0.0]]])  # 1 fruit, 1 height bin, 2 radius bins
    sums = np.array([[[[4.0, 0.0]], [[6.0, 0.0]]]])  # 2 value axes
    means = fruit_bin_means(counts, sums)
    assert means.shape == sums.shape
    np.testing.assert_allclose(means[0, :, 0, 0], [2.0, 3.0])
    assert np.isnan(means[0, :, 0, 1]).all()


def test_group_bin_summary_counts_every_fruit_once_and_applies_the_share_rule():
    from pearlatent.step0 import group_bin_summary

    # Bin 0: three fruit with values 1, 2, 9; a mean over cells would weigh fruit by size,
    # this mean does not. Bin 1: only one of three fruit has a value (below half): NaN.
    means = np.array([[[1.0, 5.0]], [[2.0, np.nan]], [[9.0, np.nan]]])
    summary = group_bin_summary(means, np.array([True, True, True]))
    assert summary["mean"][0, 0] == pytest.approx(4.0)
    assert summary["q25"][0, 0] == pytest.approx(1.5)
    assert np.isnan(summary["mean"][0, 1])

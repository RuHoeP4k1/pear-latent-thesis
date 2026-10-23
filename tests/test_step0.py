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
        "widest_radius_voxels": 24.0,
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
    # 24 cover radius 0 to 0.25 of the widest radius, 6 / 24 = 0.25).
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

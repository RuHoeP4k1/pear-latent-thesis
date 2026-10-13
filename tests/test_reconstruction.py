import numpy as np
import pytest

from pearlatent.pooling import block_mean
from pearlatent.reconstruction import (
    CORE_RADIUS_FRACTION,
    body_centre,
    body_voxel_mask,
    core_voxel_mask,
    masked_mae,
    slice_mae,
)


def _cylinder(radius=24, centre=(64, 64), z=(20, 100), shape=(128, 128, 128)):
    """A solid cylinder along the third axis, intensity 0.8, with a small empty hole inside."""
    gx, gy = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    disc = (gx - centre[0]) ** 2 + (gy - centre[1]) ** 2 <= radius**2
    vol = np.zeros(shape, dtype=np.float32)
    vol[:, :, z[0] : z[1]] = 0.8 * disc[:, :, None]
    vol[60:68, 60:68, 56:64] = 0.0  # an internal cavity
    return vol


def test_body_fills_cavity_and_stays_inside_cells():
    vol = _cylinder()
    cells = block_mean((vol > 0.1).astype(np.float32)) >= 0.5
    body = body_voxel_mask(vol, cells)
    assert body[64, 64, 60]  # the cavity counts as body
    assert not body[64, 64, 10]  # below the cylinder
    assert not body[5, 5, 60]  # background


def test_core_radius_and_axis():
    vol = _cylinder(radius=24, centre=(60, 70))
    body = vol > 0.1
    body[60:68, 60:68, 56:64] = True
    core, geometry = core_voxel_mask(body)
    assert geometry["axis_x"] == pytest.approx(60, abs=0.5)
    assert geometry["axis_y"] == pytest.approx(70, abs=0.5)
    # widest radius from the disc area is close to the true radius
    assert geometry["widest_radius_voxels"] == pytest.approx(24, abs=0.5)
    expected = CORE_RADIUS_FRACTION * geometry["widest_radius_voxels"]
    assert geometry["core_radius_voxels"] == pytest.approx(expected)
    assert core[60, 70, 50] and not core[60, 70 + 12, 50]
    assert not (core & ~body).any()  # the core lies inside the body


def test_masked_and_slice_mae():
    residual = np.zeros((4, 4, 3), dtype=np.float32)
    residual[:, :, 1] = 2.0
    mask = np.zeros_like(residual, dtype=bool)
    mask[0, 0, :2] = True
    assert masked_mae(residual, mask) == pytest.approx(1.0)
    assert np.isnan(masked_mae(residual, np.zeros_like(mask)))
    np.testing.assert_allclose(slice_mae(residual), [0.0, 2.0, 0.0])
    per_slice = slice_mae(residual, mask)
    assert per_slice[0] == 0.0 and per_slice[1] == 2.0 and np.isnan(per_slice[2])


def test_body_centre():
    body = np.zeros((10, 10, 10), dtype=bool)
    body[2:5, 6:9, 0:10] = True
    assert body_centre(body) == (3, 7, 4)  # 4.5 rounds to 4 (round half to even)

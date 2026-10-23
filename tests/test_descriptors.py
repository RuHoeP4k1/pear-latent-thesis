import numpy as np
import pytest

from pearlatent.descriptors import fruit_descriptors, largest_part, slice_extent


def _fruit_with_stalk():
    """A cylinder body (radius 24, slices 20 to 99) with a cavity, a thin stalk on top
    (slices 100 to 115) and a stray speck far away."""
    shape = (128, 128, 128)
    gx, gy = np.meshgrid(np.arange(128), np.arange(128), indexing="ij")
    disc = (gx - 64) ** 2 + (gy - 64) ** 2 <= 24**2
    vol = np.zeros(shape, dtype=np.float32)
    vol[:, :, 20:100] = 0.8 * disc[:, :, None]
    vol[60:68, 60:68, 56:64] = 0.0  # cavity of 8³ = 512 voxels
    vol[63:65, 63:65, 100:116] = 0.8  # stalk, 2 × 2 voxels, 16 slices
    vol[2:4, 2:4, 2:4] = 1.0  # speck of 8 voxels
    return vol


def test_largest_part_drops_the_speck_and_fills_holes():
    vol = _fruit_with_stalk()
    fruit = largest_part(vol > 0.1)
    assert not fruit[2, 2, 2]
    assert fruit[64, 64, 60]  # cavity filled
    assert fruit[64, 64, 110]  # stalk is part of the fruit


def test_slice_extent():
    mask = np.zeros((4, 4, 10), dtype=bool)
    mask[1, 1, 3:7] = True
    assert slice_extent(mask) == (3, 6)
    assert slice_extent(np.zeros((4, 4, 10), dtype=bool)) == (-1, -1)


def test_descriptors_of_synthetic_fruit():
    d = fruit_descriptors(_fruit_with_stalk())
    assert d["stray_voxels"] == 8
    assert d["body_length"] == 80  # slices 20 to 99
    assert d["stalk_length"] == 16  # slices 100 to 115
    # Equivalent diameter of the largest slice ≈ 2 × 24 (a digital disc of radius 24).
    assert d["widest_diameter"] == pytest.approx(48, abs=1)
    assert d["diameter_to_length"] == pytest.approx(48 / 80, abs=0.02)
    assert d["tissue_intensity_mean"] == pytest.approx(0.8)
    # The cavity is 512 of about π · 24² · 80 ≈ 145,000 body voxels.
    assert d["cavity_fraction"] == pytest.approx(512 / d["body_voxels"], rel=0.01)
    # The stalk is in the fruit, not the body.
    assert d["fruit_voxels"] > d["body_voxels"]


def test_descriptors_of_empty_volume():
    d = fruit_descriptors(np.zeros((128, 128, 128), dtype=np.float32))
    assert d["body_voxels"] == 0 and d["body_length"] == 0
    assert np.isnan(d["diameter_to_length"]) and np.isnan(d["tissue_intensity_mean"])

import numpy as np
import pytest

from pearlatent.descriptors import (
    fruit_descriptors,
    largest_flat_face,
    largest_part,
    slice_extent,
)


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


def _ball(radius=30, cut_at=None):
    """A solid ball in a 128³ grid, optionally with everything above x = cut_at removed."""
    g = np.indices((128, 128, 128)) - 64
    ball = (g**2).sum(axis=0) <= radius**2
    if cut_at is not None:
        ball[64 + cut_at :, :, :] = False
    return ball


def test_flat_face_small_on_ball_large_on_cut_ball():
    whole_patch, whole_surface = largest_flat_face(_ball())
    cut_patch, cut_surface = largest_flat_face(_ball(cut_at=10))
    # A cut at 10 voxels from the centre of a ball of radius 30 leaves a flat disc of radius
    # √(30² − 10²) ≈ 28, about 2,500 voxels.
    assert cut_patch / cut_surface > 2 * whole_patch / whole_surface
    assert cut_patch > 2000


def test_flat_face_is_reproducible():
    assert largest_flat_face(_ball(cut_at=5)) == largest_flat_face(_ball(cut_at=5))


def test_stalk_present():
    assert fruit_descriptors(_fruit_with_stalk())["stalk_present"]  # 16 slices


def test_descriptors_return_every_documented_number():
    d = fruit_descriptors(_fruit_with_stalk())
    for key in (
        "stalk_present",
        "flat_face_voxels",
        "surface_voxels",
        "flat_face_share",
    ):
        assert key in d
    # The flat ends of the synthetic cylinder are flat faces.
    assert d["flat_face_share"] > 0.1

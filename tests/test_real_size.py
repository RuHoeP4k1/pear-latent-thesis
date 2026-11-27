import numpy as np
import pytest

from pearlatent.real_size import (
    VOXEL_SIZE_MM,
    foreground_extent,
    rescaling_check,
    sizes_in_mm,
    stretch_factor,
)


def test_stretch_factor_is_128_over_length():
    assert stretch_factor(256) == pytest.approx(0.5)
    assert stretch_factor(np.array([128, 64])) == pytest.approx([1.0, 2.0])


def test_foreground_extent_counts_filled_slices():
    mask = np.zeros((10, 12, 20), dtype=bool)
    mask[2:5, 3:9, 4:18] = True
    assert foreground_extent(mask) == (3, 6, 14)


def test_rescaling_check_uniform_and_stretched():
    """A box of 20 × 30 × 64 voxels rescaled to 10 × 15 × 32: every factor 0.5, ratios 1.
    The same box rescaled to 20 × 15 × 32: axis 0 enlarged, ratio 2."""
    floor = 7000  # originals have a constant background floor, not 0
    original = np.full((24, 34, 64), floor, dtype=np.uint16)
    original[2:22, 2:32, :] = floor + 500
    uniform = np.zeros((128, 128, 128), dtype=np.float32)
    uniform[10:20, 10:25, 0:32] = 1.0
    check = rescaling_check(original, uniform)
    assert check["factor_axis2"] == pytest.approx(0.5)
    assert check["ratio_axis0_to_long"] == pytest.approx(1.0)
    assert check["ratio_axis1_to_long"] == pytest.approx(1.0)

    stretched = np.zeros((128, 128, 128), dtype=np.float32)
    stretched[10:30, 10:25, 0:32] = 1.0
    assert rescaling_check(original, stretched)["ratio_axis0_to_long"] == pytest.approx(
        2.0
    )


def test_sizes_in_mm_worked_example():
    """Module docstring: s = 128/243, 150,000 body voxels -> about 253 cm³."""
    s = np.array([128 / 243])
    out = sizes_in_mm(
        body_voxels=np.array([150_000]),
        body_length=np.array([100]),
        stalk_length=np.array([10]),
        widest_diameter=np.array([50.0]),
        stretch=s,
    )
    voxel = VOXEL_SIZE_MM * 243 / 128
    assert out["voxel_mm_128"][0] == pytest.approx(voxel)
    assert out["body_volume_cm3"][0] == pytest.approx(150_000 * voxel**3 / 1000)
    assert out["body_volume_cm3"][0] == pytest.approx(253, abs=1)
    assert out["stalk_length_mm"][0] == pytest.approx(10 * voxel)

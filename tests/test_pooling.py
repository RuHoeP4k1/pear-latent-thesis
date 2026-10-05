import numpy as np
import pytest

from pearlatent.pooling import (
    block_mean,
    cylindrical_coordinates,
    cylindrical_profile,
    fruit_mask,
    pool_grid,
    pool_grids,
    scale_minmax,
)


def _cylinder(shape=(32, 32, 32), radius=10, z0=0, z1=32):
    ii, jj, kk = np.meshgrid(*(np.arange(n) for n in shape), indexing="ij")
    c = (shape[0] - 1) / 2
    return ((ii - c) ** 2 + (jj - c) ** 2 <= radius**2) & (kk >= z0) & (kk < z1)


def test_scale_minmax_and_block_mean():
    v = np.arange(8**3, dtype=np.uint16).reshape(8, 8, 8)
    s = scale_minmax(v)
    assert s.min() == 0.0 and s.max() == 1.0 and s.dtype == np.float32
    b = block_mean(np.ones((8, 8, 8)), factor=4)
    assert b.shape == (2, 2, 2) and np.all(b == 1)
    with pytest.raises(ValueError):
        block_mean(np.ones((6, 8, 8)), factor=4)


def test_fruit_mask_needs_half_of_the_block():
    v = np.zeros((8, 8, 8), dtype=np.float32)
    v[:4, :4, :4] = 1.0  # one full block
    v[4:, 4:, 4:6] = 1.0  # half a block
    v[4:, :4, :1] = 1.0  # a quarter of a block
    m = fruit_mask(v, factor=4)
    assert m[0, 0, 0] and m[1, 1, 1] and not m[1, 0, 0]


def test_cylindrical_coordinates_of_a_cylinder():
    mask = _cylinder(z0=4, z1=28)
    h, r = cylindrical_coordinates(mask)
    assert np.isnan(h[~mask]).all() and np.isnan(r[~mask]).all()
    assert np.nanmin(h) == 0.0 and np.nanmax(h) == 1.0
    assert np.nanmin(r) == pytest.approx(0.0, abs=0.1) and np.nanmax(r) == 1.0
    # Height runs along the third axis
    assert np.nanmax(h[:, :, 4]) == 0.0 and np.nanmin(h[:, :, 27]) == 1.0


def test_cylindrical_profile_separates_core_from_skin_and_ends():
    mask = _cylinder()
    h, r = cylindrical_coordinates(mask)
    grid = np.where(mask, r, 0.0)  # value = normalised radius
    prof = cylindrical_profile(grid, mask, n_height=4, n_radius=4)
    assert prof.shape == (4, 4)
    # Increases from core to skin in every height bin
    assert np.all(np.diff(prof, axis=1) > 0)
    grid2 = np.where(mask, h, 0.0)
    prof2 = cylindrical_profile(grid2, mask, n_height=4, n_radius=4)
    assert np.all(np.diff(prof2, axis=0) > 0)


def test_pool_grid_ignores_background():
    mask = _cylinder()
    grid = np.where(mask, 2.0, -100.0)
    f = pool_grid(grid, mask, n_height=2, n_radius=2)
    assert f["mean"] == 2.0 and f["p99"] == 2.0
    assert set(f) == {"mean", "p99", "cyl_h0_r0", "cyl_h0_r1", "cyl_h1_r0", "cyl_h1_r1"}
    assert all(v == 2.0 for v in f.values())
    with pytest.raises(ValueError, match="empty"):
        pool_grid(grid, np.zeros_like(mask))


def test_pool_grids_table():
    mask = _cylinder()
    grids = np.stack([np.where(mask, 1.0, 0.0), np.where(mask, 3.0, 0.0)])
    df = pool_grids(grids, np.stack([mask, mask]), ["a", "b"], n_height=2, n_radius=2)
    assert df["fruit_id"].to_list() == ["a", "b"]
    assert df["mean"].to_list() == [1.0, 3.0]
    assert df["n_mask_cells"][0] == int(mask.sum())


def test_body_mask_removes_stalk_and_islands():
    from pearlatent.pooling import body_mask

    mask = _cylinder(z0=0, z1=24)
    mask[15:17, 15:16, 24:30] = True  # thin stalk of 2 cells per slice, attached
    mask[2, 2, 31] = True  # isolated cell
    body = body_mask(mask)
    assert body[:, :, :24].sum() == mask[:, :, :24].sum()
    assert not body[:, :, 24:].any()


def test_body_mask_fills_internal_cavity():
    from pearlatent.pooling import body_mask

    mask = _cylinder(z0=0, z1=24)
    mask[14:18, 14:18, 10:14] = False  # enclosed cavity
    assert body_mask(mask)[14:18, 14:18, 10:14].all()

import numpy as np
import polars as pl

from pearlatent.ipls import (
    forward_ipls,
    height_bin_labels,
    shell_labels,
    square_ring_labels,
)


def test_shells_inside_the_radius_have_about_equal_size():
    labels, boundaries = shell_labels(n_shells=10, max_radius=10.0)
    counts = np.bincount(labels.ravel())
    assert len(counts) == 11
    inner = counts[:10]
    assert inner.max() / inner.min() < 1.5
    assert boundaries[0] == 0 and boundaries[-1] == 10.0
    # Every column has one shell over its whole height.
    assert (labels == labels[:, :, :1]).all()


def test_forward_ipls_finds_the_shell_that_carries_the_label():
    rng = np.random.default_rng(0)
    labels, _ = shell_labels(n_shells=4, max_radius=10.0, grid=8)
    flat = labels.ravel()
    boxes = np.repeat([f"B{k:02d}" for k in range(12)], 20)
    y = rng.integers(0, 2, boxes.size)
    x = rng.normal(size=(boxes.size, flat.size))
    x[:, flat == 0] += 1.0 * y[:, None]  # only the innermost shell carries the label
    folds = pl.DataFrame(
        {
            "repeat": [1] * 12,
            "fold": [k // 3 + 1 for k in range(12)],
            "box": [f"B{k:02d}" for k in range(12)],
        }
    )
    alone, path = forward_ipls(x, y, boxes, folds, labels)
    assert alone.sort("rmsecv")["shell"][0] == 0
    assert path["added"][0] == 0
    assert path.height == alone.height


def test_square_rings_are_one_cell_thick_from_edge_to_centre():
    rings = square_ring_labels()
    counts = np.bincount(rings[:, :, 0].ravel())
    # Ring k is the edge of a (32 - 2k) square: (32 - 2k)^2 - (30 - 2k)^2 columns.
    assert counts.tolist() == [
        (32 - 2 * k) ** 2 - max(30 - 2 * k, 0) ** 2 for k in range(16)
    ]
    assert counts[0] == 124 and counts[15] == 4
    assert (rings == rings[:, :, :1]).all()


def test_height_bins_cover_only_the_selected_cells():
    selected = square_ring_labels() >= 14
    bins = height_bin_labels(selected, slices_per_bin=2)
    assert (bins[~selected] == -1).all()
    assert sorted(np.unique(bins[selected])) == list(range(16))
    assert (bins[:, :, 0][selected[:, :, 0]] == 0).all()
    assert (bins[:, :, 31][selected[:, :, 31]] == 15).all()

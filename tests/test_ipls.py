import numpy as np
import polars as pl

from pearlatent.ipls import forward_ipls, shell_labels


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

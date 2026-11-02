"""Interval PLS (iPLS) on cylindrical shells around the stem-calyx axis (Ruben, 2 November 2026).

Forward iPLS after Nørgaard et al. (2000, *Applied Spectroscopy* 54, 413-419), with intervals
that are shells of the 32³ grid instead of spectral windows:

- `shell_labels`: every cell gets the number of its shell. A shell is a hollow cylinder around
  the grid axis (the stem-calyx axis after Hugo's alignment, the third array axis), over the
  full height. Shells inside `max_radius` hold (nearly) the same number of cells, so an inner
  shell is not smaller than an outer one; the last label collects everything outside
  `max_radius` (mostly cells outside the fruit). The shells are fixed on the grid, the same for
  every fruit, so a variable means the same cell in every fruit (unfolded PLS needs that).
- `interval_rmsecv`: the mean RMSECV curve (1 to 15 LVs) of PLS on a subset of cells, with the
  20 box partitions of design A.
- `forward_ipls`: start with no shell; at each step add the shell that gives the lowest
  RMSECV (lowest point of the curve, Ruben's rule of 2 November); continue until all shells
  are in, so the whole path can be plotted. The selected set is the step with the lowest RMSECV.

The RMSECV of the selected set is slightly optimistic: the shells are chosen on the same
cross-validation that scores them. The sealed test boxes give the independent check.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from pearlatent.plsda import (
    MAX_COMPONENTS,
    cross_validated_predictions,
    mean_squared_errors,
)

GRID = 32
N_SHELLS = 10
# Fruit bodies reach up to about 10 cells from the grid axis in the open 2526 fruit (columns
# with body in at least 1 % of fruit: radius up to 9.9 cells, measured 2 November 2026).
MAX_RADIUS = 10.0


def column_radius(grid: int = GRID) -> np.ndarray:
    """Distance of every (x, y) column centre to the grid axis, in cells: array (grid, grid)."""
    centre = (grid - 1) / 2
    x, y = np.meshgrid(np.arange(grid), np.arange(grid), indexing="ij")
    return np.hypot(x - centre, y - centre)


def shell_labels(
    n_shells: int = N_SHELLS, max_radius: float = MAX_RADIUS, grid: int = GRID
) -> tuple[np.ndarray, np.ndarray]:
    """Shell number of every cell, and the shell boundaries in cells.

    Out: labels (grid, grid, grid) with values 0 (innermost) to n_shells (outside
    `max_radius`), and the n_shells + 1 boundary radii. The boundaries are quantiles of the
    radii of the columns inside `max_radius`, so each shell gets about the same number of
    columns; columns at exactly the same radius stay together.
    """
    radius = column_radius(grid)
    inside = radius[radius <= max_radius]
    boundaries = np.quantile(inside, np.linspace(0, 1, n_shells + 1))
    boundaries[0], boundaries[-1] = 0.0, max_radius
    column_label = np.searchsorted(boundaries[1:-1], radius, side="right")
    column_label[radius > max_radius] = n_shells
    labels = np.repeat(column_label[:, :, None], grid, axis=2)
    return labels, boundaries


def square_ring_labels(grid: int = GRID) -> np.ndarray:
    """Layer number of every cell: square rings of whole cells around the grid axis.

    Ruben, 2 November 2026: layers of voxels, no computed radius. Ring 0 is the outer edge of
    the 32 × 32 cross-section (the 124 columns of the 32 × 32 square that are not in the
    30 × 30 square inside it); ring 1 is the edge of that 30 × 30 square; and so on to ring 15,
    the 2 × 2 columns at the centre. Each ring is one cell thick on every side and runs over the
    full height. Out: labels (grid, grid, grid), values 0 to grid / 2 - 1.
    """
    index = np.arange(grid)
    # Number of cells between a column and the nearest outer edge of the grid.
    from_edge = np.minimum(index, grid - 1 - index)
    ring = np.minimum(from_edge[:, None], from_edge[None, :])
    return np.repeat(ring[:, :, None], grid, axis=2)


def height_bin_labels(
    selected_cells: np.ndarray, slices_per_bin: int = 2, grid: int = GRID
) -> np.ndarray:
    """Height bin of every selected cell; -1 for cells not selected.

    In: a boolean array (grid, grid, grid), for example the cells of the rings chosen in the
    first stage. Out: labels (grid, grid, grid): bin k holds the selected cells in slices
    k * slices_per_bin to (k + 1) * slices_per_bin - 1 along the third axis (the stem-calyx
    axis, calyx end at slice 0).
    """
    height = np.arange(grid) // slices_per_bin
    labels = np.broadcast_to(height[None, None, :], (grid, grid, grid)).copy()
    labels[~selected_cells] = -1
    return labels


def interval_rmsecv(
    x: np.ndarray, y: np.ndarray, boxes: np.ndarray, folds: pl.DataFrame
) -> np.ndarray:
    """Mean RMSECV for 1 to 15 LVs: sqrt(mean PRESS over the repeats / n). Centring only.

    PLS cannot extract more LVs than the rank of the centred training data (a layer of the
    CT where only a few fruit reach has a rank below 15). The number of LVs is therefore
    capped at the smallest rank over the training sets of all folds, minus one for safety;
    the curve is padded with infinity beyond that cap, so those LVs are never chosen.
    """
    n_components = min(
        MAX_COMPONENTS, x.shape[1], _smallest_training_rank(x, boxes, folds) - 1
    )
    curve = np.full(MAX_COMPONENTS, np.inf)
    if n_components < 1:
        curve[:] = float(np.std(y))
        return curve
    predictions = cross_validated_predictions(
        x, y, boxes, folds, scale=False, max_components=n_components
    )
    curve[:n_components] = np.sqrt(
        mean_squared_errors(predictions, y).sum(axis=0) / len(y)
    )
    return curve


def _smallest_training_rank(
    x: np.ndarray, boxes: np.ndarray, folds: pl.DataFrame
) -> int:
    """Smallest rank of the centred training data over all folds of all repeats.

    Only computed when the rank of all fruit together is low; otherwise every training set
    of 9 boxes (about 235 fruit) has far more than 15 dimensions and the overall rank is
    returned.
    """
    overall = int(np.linalg.matrix_rank(x - x.mean(axis=0)))
    if overall > 2 * MAX_COMPONENTS:
        return overall
    boxes = np.asarray(boxes)
    smallest = overall
    for (repeat, fold), _ in folds.group_by(["repeat", "fold"]):
        held_out_boxes = folds.filter(
            (pl.col("repeat") == repeat) & (pl.col("fold") == fold)
        )["box"]
        train = x[~np.isin(boxes, held_out_boxes.to_list())]
        smallest = min(smallest, int(np.linalg.matrix_rank(train - train.mean(axis=0))))
    return smallest


def forward_ipls(
    x: np.ndarray,
    y: np.ndarray,
    boxes: np.ndarray,
    folds: pl.DataFrame,
    labels: np.ndarray,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Each shell alone, and the forward path.

    In: features (fruit, cells) in the same cell order as `labels.ravel()`. Out: (alone, path).
    `alone`: one row per shell with its number of cells, the LVs at the lowest RMSECV and that
    RMSECV. `path`: one row per step with the shell added, the shells in the model, the LVs and
    the RMSECV after the step.
    """
    flat = labels.ravel()
    # Cells with a negative label belong to no interval and are never used.
    shells = np.unique(flat[flat >= 0])
    curves = {}
    varies = x.std(axis=0) > 0

    def best(selected: tuple[int, ...]) -> tuple[int, float]:
        if selected not in curves:
            # Cells that have the same value in every fruit (air around the fruit in the CT)
            # carry nothing and make PLS divide by zero; they are left out. This uses no
            # labels. If nothing is left, the curve is that of predicting the mean.
            columns = np.isin(flat, selected) & varies
            if columns.any():
                curves[selected] = interval_rmsecv(x[:, columns], y, boxes, folds)
            else:
                curves[selected] = np.full(MAX_COMPONENTS, float(np.std(y)))
        curve = curves[selected]
        return int(np.argmin(curve)) + 1, float(curve.min())

    alone_rows = []
    for shell in shells:
        n_lv, rmsecv = best((int(shell),))
        alone_rows.append(
            {
                "shell": int(shell),
                "n_cells": int((flat == shell).sum()),
                "n_lv": n_lv,
                "rmsecv": rmsecv,
            }
        )

    selected: tuple[int, ...] = ()
    path_rows = []
    remaining = [int(s) for s in shells]
    while remaining:
        trials = {s: best(tuple(sorted((*selected, s)))) for s in remaining}
        added = min(trials, key=lambda s: trials[s][1])
        selected = tuple(sorted((*selected, added)))
        remaining.remove(added)
        n_lv, rmsecv = trials[added]
        path_rows.append(
            {
                "step": len(selected),
                "added": added,
                "shells": list(selected),
                "n_lv": n_lv,
                "rmsecv": rmsecv,
            }
        )
    return pl.DataFrame(alone_rows), pl.DataFrame(path_rows)

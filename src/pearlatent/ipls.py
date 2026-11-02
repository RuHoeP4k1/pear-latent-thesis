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


def interval_rmsecv(
    x: np.ndarray, y: np.ndarray, boxes: np.ndarray, folds: pl.DataFrame
) -> np.ndarray:
    """Mean RMSECV for 1 to 15 LVs: sqrt(mean PRESS over the repeats / n). Centring only."""
    n_components = min(MAX_COMPONENTS, x.shape[1])
    predictions = cross_validated_predictions(
        x, y, boxes, folds, scale=False, max_components=n_components
    )
    return np.sqrt(mean_squared_errors(predictions, y).sum(axis=0) / len(y))


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
    shells = np.unique(flat)
    curves = {}

    def best(selected: tuple[int, ...]) -> tuple[int, float]:
        if selected not in curves:
            columns = np.isin(flat, selected)
            curves[selected] = interval_rmsecv(x[:, columns], y, boxes, folds)
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

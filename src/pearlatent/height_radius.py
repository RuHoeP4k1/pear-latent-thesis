"""Height-radius features: a 32³ grid averaged over the angle about the stem-calyx axis.

Reopened as a model row by Ruben on 7 November 2026 (the decision of 10 October had dropped the
cylindrical bins); see `docs/methods_plsda.md` section 10.

Why: rotation about the stem-calyx axis is not aligned between fruit (CLAUDE.md section 6), so
cell (x, y, z) of the grid lies at a different angle in different fruit. The mean over all
body cells of one (height, radius) bin does not depend on that angle. 20 height bins × 8
radius bins = 160 features per fruit, the same bins as the maps of notebooks 06 and 07.

Radius per slice (`latent_map_sums(..., "slice")`): distance to the centre of the body cells
of the same slice divided by the outermost body cell of that slice, so a radius bin is a
relative depth below the skin.

Empty bins: the body spans 21 to 32 slices, so each of the 20 height bins holds at least one
slice (no fruit of the 315 has an empty height bin, checked 9 November 2026). A slice has few
cells, though, so it may not reach every radius bin, most often near the axis (median 9, at most
22 of 160 per fruit in the 315 fruit). PLS needs a value in every bin, so an empty bin takes the mean of the
nearest filled bins of the same fruit (distance in bin steps). No label and no other fruit is
used, so the filling cannot leak between boxes.
"""

from __future__ import annotations

import numpy as np

from pearlatent.step0 import (
    N_LATENT_HEIGHT_BINS,
    N_LATENT_RADIUS_BINS,
    fruit_bin_means,
    latent_map_sums,
)

RADIUS_METHOD = "slice"  # Ruben, 7 November 2026


def fill_empty_bins(means: np.ndarray) -> tuple[np.ndarray, int]:
    """Fill NaN bins of one fruit's (height, radius) map from its nearest filled bins.

    In: a 2D map with NaN where the fruit has no cell. Out: the filled map and the number of
    bins that were filled. Distance is measured in bin steps (height and radius alike); when
    several filled bins are equally near, their mean is used. Raises if the map is all NaN.
    """
    means = np.asarray(means, dtype=float)
    empty = np.isnan(means)
    if empty.all():
        raise ValueError("no filled bin to copy from")
    filled = means.copy()
    filled_h, filled_r = np.nonzero(~empty)
    for h, r in zip(*np.nonzero(empty), strict=True):
        distance = np.hypot(filled_h - h, filled_r - r)
        nearest = distance == distance.min()
        filled[h, r] = means[filled_h[nearest], filled_r[nearest]].mean()
    return filled, int(empty.sum())


def height_radius_features(
    grids: np.ndarray,
    body_cells: np.ndarray,
    n_height: int = N_LATENT_HEIGHT_BINS,
    n_radius: int = N_LATENT_RADIUS_BINS,
) -> tuple[np.ndarray, np.ndarray]:
    """160 height-radius means per fruit.

    In: grids (fruit, 32, 32, 32), the matching 32³ body cells (fruit, 32, 32, 32).
    Out: features (fruit, n_height * n_radius), ordered height first (feature index =
    n_radius * height bin + radius bin, as the spectrum of notebook 07), and the number of
    filled bins per fruit.
    """
    features = np.empty((len(grids), n_height * n_radius), dtype=np.float32)
    n_filled = np.empty(len(grids), dtype=int)
    for k in range(len(grids)):
        counts, sums = latent_map_sums(
            np.asarray(grids[k]),
            np.asarray(body_cells[k]),
            RADIUS_METHOD,
            n_height,
            n_radius,
        )
        means = fruit_bin_means(counts[None], sums[None])[0]
        filled, n_filled[k] = fill_empty_bins(means)
        features[k] = filled.ravel()
    return features, n_filled

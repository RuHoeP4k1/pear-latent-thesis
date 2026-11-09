"""Spread of a 32³ grid over the interior of the fruit body (notebook 15; Ruben, 9 November 2026).

Why the interior: the outermost layer of body cells straddles skin and air, so their values are
partly background. That layer varies a lot in every fruit, healthy or not, and dominates an SD
taken over all body cells. Removing it leaves the tissue inside.

Choice made after looking (stated in the notebook): in a scratch run on 9 November the CT SD
over all body cells did not separate grade 0 from grade 1 (AUC 0.53), the SD without the outer
layer did (0.82), and removing 2 or 3 layers gave the same (0.82). One layer is used.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_erosion

# Layers of body cells removed from the outside (one cell is about 2.5 mm).
N_EDGE_LAYERS = 1


def interior_cells(body_cells: np.ndarray, n_layers: int = N_EDGE_LAYERS) -> np.ndarray:
    """The body cells that remain after removing `n_layers` outer layers.

    In: one fruit's 32³ body cells (boolean). Out: a boolean grid of the same shape. A layer is
    every body cell that touches a non-body cell across a face (6 neighbours,
    `scipy.ndimage.binary_erosion` with its default structure). `n_layers` = 0 returns the body.
    """
    body = np.asarray(body_cells, dtype=bool)
    if n_layers == 0:
        return body
    return binary_erosion(body, iterations=n_layers)


def interior_sd(
    grids: np.ndarray, body_cells: np.ndarray, n_layers: int = N_EDGE_LAYERS
) -> np.ndarray:
    """SD of each fruit's grid values over its interior cells, one number per fruit.

    In: grids (fruit, 32, 32, 32) and the matching body cells. Out: an array (fruit,), SD
    divided by n. Raises when a fruit has fewer than 2 interior cells.
    """
    out = np.empty(len(grids))
    for k in range(len(grids)):
        inside = interior_cells(body_cells[k], n_layers)
        if inside.sum() < 2:
            raise ValueError(f"fruit {k}: fewer than 2 interior cells")
        out[k] = np.asarray(grids[k], dtype=float)[inside].std()
    return out

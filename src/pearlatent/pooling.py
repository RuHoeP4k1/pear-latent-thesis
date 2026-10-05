"""Pooling of 32³ grids (latents or raw CT averaged to 32³) into a feature table.

Step 2 of the pipeline (CLAUDE.md section 5): array arithmetic only, no labels.

Grid geometry. A latent cell covers about 4 voxels along each axis (CLAUDE.md section 3), so a
32³ grid lines up with blocks of 4³ voxels of the 128³ volume. The fruit mask at 32³ is computed
from the volume with the same blocks, and is used for every grid of that fruit (raw CT, latent
`mu`, the 512-channel layer).

Why pool inside the mask. The number of background cells depends on fruit shape and, in 2526,
on stalk length, and both differ between orchards (notebooks 02 and 03, 5 October 2026). A mean
over all 32³ cells would mix these into every feature; a mean over fruit cells does not.

Orientation (notebook 02): the long axis is the third array axis in every volume, and the wide
(calyx) end is at the low end of that axis in every volume. The 2526 volumes include the stalk
at the high end; `body_mask` removes it.
"""

from __future__ import annotations

import numpy as np
import polars as pl

GRID = 32
FACTOR = 4
# Same threshold as the volume quality check (pearlatent.volume_qc.DEFAULT_THRESHOLD)
MASK_THRESHOLD = 0.1
# A cell is fruit when at least this fraction of its 4³ voxels is fruit
MASK_MIN_FRACTION = 0.5
# Slices of the fruit body have at least this many cells; thinner slices are stalk
MIN_BODY_CELLS = 4
N_HEIGHT = 8
N_RADIUS = 4


def scale_minmax(volume: np.ndarray) -> np.ndarray:
    """Min-max scale to [0, 1] in float32, as Hugo's `utils/volumes.py` `normalize_minmax`."""
    v = np.asarray(volume, dtype=np.float32)
    lo, hi = float(v.min()), float(v.max())
    if hi - lo < 1e-8:
        return np.zeros_like(v)
    return (v - lo) / (hi - lo)


def block_mean(volume: np.ndarray, factor: int = FACTOR) -> np.ndarray:
    """Average non-overlapping blocks of factor³ voxels (128³ -> 32³ for factor 4)."""
    v = np.asarray(volume, dtype=np.float32)
    if any(n % factor for n in v.shape):
        raise ValueError(f"shape {v.shape} is not divisible by {factor}")
    a, b, c = (n // factor for n in v.shape)
    return v.reshape(a, factor, b, factor, c, factor).mean(axis=(1, 3, 5))


def fruit_mask(
    scaled: np.ndarray,
    threshold: float = MASK_THRESHOLD,
    factor: int = FACTOR,
    min_fraction: float = MASK_MIN_FRACTION,
) -> np.ndarray:
    """Boolean mask at grid resolution: cells where at least `min_fraction` of the voxels are fruit."""
    return block_mean((scaled > threshold).astype(np.float32), factor) >= min_fraction


def body_mask(
    mask: np.ndarray, min_cells_per_slice: int = MIN_BODY_CELLS
) -> np.ndarray:
    """The fruit body: the largest connected part of `mask`, without thin slices along the third axis.

    The 2526 volumes include the stalk, and the stalk is counted in the 128-voxel length
    (notebook 03, 5 October 2026). At 32³ the stalk leaves isolated cells or slices of one to
    three cells; this removes them, so height runs over the body only. Enclosed holes are
    filled, so internal cavities (air, below the intensity threshold) count as fruit body and
    reach the pooled features.
    """
    from scipy import ndimage

    m = np.asarray(mask, dtype=bool)
    labels, n = ndimage.label(m)
    if n == 0:
        return m.copy()
    largest = labels == (np.argmax(np.bincount(labels.ravel())[1:]) + 1)
    thin = largest.sum(axis=(0, 1)) < min_cells_per_slice
    largest[:, :, thin] = False
    return ndimage.binary_fill_holes(largest)


def cylindrical_coordinates(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalised height and radius of every cell, NaN outside the mask.

    Height: position along the third axis between the first (0) and last (1) masked slice.
    Radius: distance in the plane of the first two axes from the centroid of the masked cells of
    the same slice, divided by the largest such distance in that slice (so the skin is near 1 at
    every height and the core near 0). A slice with a single masked cell gets radius 0.
    """
    m = np.asarray(mask, dtype=bool)
    height = np.full(m.shape, np.nan, dtype=np.float32)
    radius = np.full(m.shape, np.nan, dtype=np.float32)
    slices = np.flatnonzero(m.any(axis=(0, 1)))
    if len(slices) == 0:
        return height, radius
    first, last = slices[0], slices[-1]
    span = max(last - first, 1)
    ii, jj = np.meshgrid(np.arange(m.shape[0]), np.arange(m.shape[1]), indexing="ij")
    for k in slices:
        s = m[:, :, k]
        ci, cj = ii[s].mean(), jj[s].mean()
        d = np.hypot(ii - ci, jj - cj)
        dmax = d[s].max()
        radius[:, :, k] = np.where(s, d / dmax if dmax > 0 else 0.0, np.nan)
        height[:, :, k] = np.where(s, (k - first) / span, np.nan)
    return height, radius


def cylindrical_profile(
    grid: np.ndarray,
    mask: np.ndarray,
    n_height: int = N_HEIGHT,
    n_radius: int = N_RADIUS,
) -> np.ndarray:
    """Mean of `grid` in each (height bin, radius bin), averaged over angle; NaN for empty bins.

    Height bin 0 is the low end of the third axis (the calyx end, notebook 02).
    """
    height, radius = cylindrical_coordinates(mask)
    m = np.asarray(mask, dtype=bool)
    h = np.minimum((height[m] * n_height).astype(int), n_height - 1)
    r = np.minimum((radius[m] * n_radius).astype(int), n_radius - 1)
    values = np.asarray(grid, dtype=np.float64)[m]
    sums = np.zeros((n_height, n_radius))
    counts = np.zeros((n_height, n_radius))
    np.add.at(sums, (h, r), values)
    np.add.at(counts, (h, r), 1)
    with np.errstate(invalid="ignore"):
        return sums / np.where(counts > 0, counts, np.nan)


def pool_grid(
    grid: np.ndarray,
    mask: np.ndarray,
    n_height: int = N_HEIGHT,
    n_radius: int = N_RADIUS,
) -> dict[str, float]:
    """Features of one grid: mean and 99th percentile inside the mask, and the cylindrical profile."""
    inside = np.asarray(grid, dtype=np.float64)[np.asarray(mask, dtype=bool)]
    if inside.size == 0:
        raise ValueError("empty mask")
    features = {"mean": float(inside.mean()), "p99": float(np.percentile(inside, 99))}
    profile = cylindrical_profile(grid, mask, n_height, n_radius)
    for i in range(n_height):
        for j in range(n_radius):
            features[f"cyl_h{i}_r{j}"] = float(profile[i, j])
    return features


def pool_grids(
    grids: np.ndarray,
    masks: np.ndarray,
    fruit_ids: list[str],
    n_height: int = N_HEIGHT,
    n_radius: int = N_RADIUS,
) -> pl.DataFrame:
    """Feature table, one row per fruit: `fruit_id`, `n_mask_cells`, `n_mask_slices` (slices
    along the third axis that contain mask cells, the body length in cells when `masks` are
    body masks), then the `pool_grid` features."""
    if not (len(grids) == len(masks) == len(fruit_ids)):
        raise ValueError("grids, masks and fruit_ids differ in length")
    rows = [
        {
            "fruit_id": f,
            "n_mask_cells": int(np.asarray(m).sum()),
            "n_mask_slices": int(np.asarray(m).any(axis=(0, 1)).sum()),
        }
        | pool_grid(g, m, n_height, n_radius)
        for g, m, f in zip(grids, masks, fruit_ids, strict=True)
    ]
    return pl.DataFrame(rows)

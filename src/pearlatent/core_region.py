"""Core region for the core-only rows (Ruben, 20 November 2026; HANDOFF question 57).

Why: the latent full-grid score follows stalk length and fruit size on healthy fruit (check C),
and the counterfactual direction added a stalk (2526_C23). A region that holds only the lower
part of the body leaves out the neck and the stalk by construction. It is a prior choice from
the rubric and pear biology, made before any model on it was fitted, not a selection on labels.

Region of one fruit: the body cells of its 32³ grid (`masks.npy`, from
`pearlatent.pooling.body_mask`: stalk removed, enclosed holes filled) whose height lies in
[HEIGHT_LOW, HEIGHT_HIGH). Height is measured over the body, as in
`pearlatent.step0.latent_map_sums`: 0 at the calyx end (low end of the third axis), 1 at the
stalk end. All radii are kept.

Bounds (drawn on 4 open 2526 fruit and approved by Ruben, 20 November 2026): the seed chambers
lie between about 0.15 and 0.45 of the body height and within 0.33 of the slice radius. 0.05
drops the bottom slice with the calyx eye; 0.60 leaves margin above the seed chambers and keeps
the neck out. A radius bound of 0.5 was considered and dropped: in the two browning 3 fruit
drawn, most darker tissue lay between radius 0.5 and the skin.

Full grid with a different region per fruit: PLS needs the same 32,768 columns for every fruit,
but the region covers different cells in each fruit. Cells outside a fruit's region are set to
`fill`, the mean of that cell over the training fruit that have it inside their region. The
column mean of the filled training data is then exactly `fill`, so after centring every cell
outside a fruit's region is 0 and adds nothing to that fruit's PLS score. `fill` is fitted,
so it is computed on the training fruit of every split only, as any other fitted step.

A fixed cell set (cells inside the region for at least 95 % of fruit) was considered and
dropped (20 November 2026): on the 315 open 2526 fruit it kept a median 57 % of a fruit's own
region, 48 % for browning 3 against 61 % for grade 0 (correlation with body voxels −0.94), so
large fruit lost their outer flesh. The fill does not remove size: a large fruit still has more
region cells, so more terms in its score. Check C on the core-only rows measures what remains.
"""

from __future__ import annotations

import numpy as np

HEIGHT_LOW = 0.05  # fraction of body height, 0 = calyx end
HEIGHT_HIGH = 0.60


def body_height(body: np.ndarray) -> np.ndarray:
    """Height of every cell over the body: 0 at the first body slice, 1 at the stalk end.

    In: one fruit's 32³ body cells (boolean). Out: an array of the same shape with the height
    of each cell's slice; cells outside the body get NaN. Same definition as
    `step0.latent_map_sums`: (slice - first body slice) / (number of body slices), so the last
    body slice gets a height just below 1. Raises when the body is empty.
    """
    body = np.asarray(body, dtype=bool)
    occupied = np.flatnonzero(body.any(axis=(0, 1)))
    if occupied.size == 0:
        raise ValueError("the body mask is empty")
    low, high = int(occupied[0]), int(occupied[-1])
    slice_height = (np.arange(body.shape[2]) - low) / (high - low + 1)
    height = np.broadcast_to(slice_height, body.shape)
    return np.where(body, height, np.nan)


def core_region(
    body: np.ndarray, low: float = HEIGHT_LOW, high: float = HEIGHT_HIGH
) -> np.ndarray:
    """Core region of one fruit: body cells with height in [low, high), all radii.

    In: the fruit's 32³ body cells. Out: a boolean 32³ array.
    """
    height = body_height(body)
    # NaN (outside the body) fails both comparisons, so those cells are never in the region.
    with np.errstate(invalid="ignore"):
        return (height >= low) & (height < high)


def core_regions(bodies: np.ndarray, **bounds) -> np.ndarray:
    """`core_region` for every fruit: (fruit, 32, 32, 32) in, the same shape out."""
    return np.stack([core_region(body, **bounds) for body in bodies])


def fit_fill(x_train: np.ndarray, regions_train: np.ndarray) -> np.ndarray:
    """Fill value per cell from the training fruit.

    In: x_train (fruit, cells) and the matching regions (fruit, cells) or (fruit, 32, 32, 32).
    Out: one value per cell: the mean of the cell over the training fruit that have it inside
    their region; 0 where no training fruit has it (the column is then constant after
    `apply_fill`, so PLS gives it weight 0).
    """
    x_train = np.asarray(x_train, dtype=float)
    inside = np.asarray(regions_train, dtype=bool).reshape(x_train.shape)
    n_inside = inside.sum(axis=0)
    total = np.where(inside, x_train, 0.0).sum(axis=0)
    return np.where(n_inside > 0, total / np.maximum(n_inside, 1), 0.0)


def apply_fill(x: np.ndarray, regions: np.ndarray, fill: np.ndarray) -> np.ndarray:
    """`x` with every cell outside each fruit's region replaced by `fill` of that cell.

    In: x (fruit, cells), regions (fruit, cells) or (fruit, 32, 32, 32), and `fill` from
    `fit_fill` on the training fruit. Out: a new array (fruit, cells); `x` is not changed.
    Cells inside a fruit's region keep their own values.
    """
    x = np.asarray(x, dtype=float)
    inside = np.asarray(regions, dtype=bool).reshape(x.shape)
    return np.where(inside, x, np.asarray(fill, dtype=float)[None, :])

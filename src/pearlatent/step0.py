"""Step 0 measures on the saved central planes (charter question 1; notebook 06).

Definitions fixed on 23 October 2026, before any of these numbers was computed.

Source: `planes.npy` of the encoding run, shape (fruit, 2, 3, 128, 128): CT (index 0) and
reconstruction `decode(mu)` (index 1) in the planes through the body centre. Plane 0 holds
the first array axis fixed (axes y, z), plane 1 the second (axes x, z), plane 2 the third
(the cross-section). Planes 0 and 1 are long sections that contain the stem-calyx axis, so
they cross the core over its whole height; only these two are used here.

- **Body on a plane**: the 32³ body cells of the fruit (`masks.npy`) cut at the plane and
  expanded to voxels, intersected with CT voxels above `MASK_THRESHOLD`, enclosed holes filled
  in two dimensions. The two-dimensional version of `reconstruction.body_voxel_mask`.
- **Core on a plane**: body voxels whose distance to the stem-calyx axis is at most the core
  radius (`core_radius_voxels` of `reconstruction.parquet`: one third of the widest radius).
- **Core intensity**: the mean CT value and the mean reconstruction value over the core voxels
  of both long sections together.
- **Contrast retention of grade g** (computed in notebook 06 from the per-fruit numbers):
  [mean core reconstruction (grade g) - mean core reconstruction (grade 0)] divided by
  [mean core CT (grade g) - mean core CT (grade 0)], means over the fruit of each group.
  1 = the reconstruction keeps the full difference from healthy fruit; 0 = the reconstruction
  of a damaged core looks like a healthy core. Numerator and denominator are reported too.
- **Height-radius map**: every body voxel of the two long sections gets a height (0 at the
  calyx end of the body, 1 at the stalk end) and a radius (distance to the axis divided by the
  distance of the outermost body voxel of the two sections, so 0 is the axis and 1 the skin
  where the fruit is widest; 27 October 2026, before: divided by the widest radius
  sqrt(largest slice area / pi), which put skin voxels of slices that are not round beyond 1). Sums of CT, reconstruction and signed residual (reconstruction - CT) are
  collected in `N_HEIGHT_BINS` × `N_RADIUS_BINS` bins, so group means can be formed by adding
  sums and counts over fruit. Averaging over the two sections and over fruit removes the free
  rotation about the axis.
"""

from __future__ import annotations

import numpy as np

from pearlatent.pooling import FACTOR, MASK_THRESHOLD

# Bins of the height-radius maps. Height 0 to 1 over the body, radius 0 to 1 (1 = the
# outermost body voxel or cell), in every map of this module.
N_HEIGHT_BINS = 32
N_RADIUS_BINS = 24
MAX_RADIUS = 1.0


def radius_bin(radius: np.ndarray, n_radius: int) -> np.ndarray:
    """Bin index of radii in [0, 1]: n_radius equal bins; radius 1 (the outermost voxel or
    cell itself) goes into the last bin."""
    return np.minimum((radius * n_radius).astype(int), n_radius - 1)


def body_on_plane(ct_plane: np.ndarray, body_cells: np.ndarray, plane: int, index: int):
    """Body voxels on one long section (128 × 128 boolean).

    In: the CT plane, the 32³ body cells, which plane (0 or 1) and the voxel index at which
    that axis is held fixed. Out: the body on the plane, holes filled.
    """
    from scipy import ndimage

    cell_index = index // FACTOR
    cells = body_cells[cell_index] if plane == 0 else body_cells[:, cell_index]
    cells_as_voxels = cells.repeat(FACTOR, 0).repeat(FACTOR, 1)
    return ndimage.binary_fill_holes((ct_plane > MASK_THRESHOLD) & cells_as_voxels)


def body_slice_range(body_cells: np.ndarray) -> tuple[int, int]:
    """First and last voxel along the stem-calyx axis covered by body cells."""
    occupied = np.flatnonzero(body_cells.any(axis=(0, 1)))
    return int(occupied[0]) * FACTOR, int(occupied[-1]) * FACTOR + FACTOR - 1


def plane_measures(fruit_planes: np.ndarray, body_cells: np.ndarray, geometry: dict):
    """Core intensities and height-radius sums of one fruit.

    In: `fruit_planes` (2, 3, 128, 128) from `planes.npy`; the fruit's 32³ body cells;
    `geometry` with plane_x, plane_y, axis_x, axis_y and core_radius_voxels (one row of
    `reconstruction.parquet`).
    Out: (numbers, sums) where numbers holds core_ct, core_reconstruction and n_core_voxels,
    and sums is an array (4, N_HEIGHT_BINS, N_RADIUS_BINS) of voxel counts and the sums of
    CT, reconstruction and signed residual.
    """
    low, high = body_slice_range(body_cells)
    sums = np.zeros((4, N_HEIGHT_BINS, N_RADIUS_BINS))
    core_ct, core_rec = [], []
    # Plane 0 holds x fixed at plane_x: its first axis is y, so the distance to the axis is
    # |y - axis_y|. Plane 1 holds y fixed: the distance is |x - axis_x|.
    sections = [
        (0, geometry["plane_x"], geometry["axis_y"]),
        (1, geometry["plane_y"], geometry["axis_x"]),
    ]
    across, along = np.meshgrid(np.arange(128), np.arange(128), indexing="ij")
    height = (along - low) / (high - low + 1)
    # First the body and the distance to the axis on both sections, because the radius is
    # divided by the outermost body voxel of the two together.
    bodies, distances = [], []
    for plane, index, axis_position in sections:
        bodies.append(body_on_plane(fruit_planes[0, plane], body_cells, plane, index))
        distances.append(np.abs(across - axis_position))
    outermost = max(
        float(d[b].max()) for b, d in zip(bodies, distances, strict=True) if b.any()
    )

    for (plane, _, _), body, distance in zip(sections, bodies, distances, strict=True):
        ct = fruit_planes[0, plane]
        rec = fruit_planes[1, plane]
        core = body & (distance <= geometry["core_radius_voxels"])
        core_ct.append(ct[core])
        core_rec.append(rec[core])

        radius = distance / outermost
        inside = body & (height >= 0) & (height < 1)
        h_bin = (height[inside] * N_HEIGHT_BINS).astype(int)
        r_bin = radius_bin(radius[inside], N_RADIUS_BINS)
        for k, values in enumerate(
            [np.ones(inside.sum()), ct[inside], rec[inside], rec[inside] - ct[inside]]
        ):
            np.add.at(sums[k], (h_bin, r_bin), values)

    core_ct = np.concatenate(core_ct)
    core_rec = np.concatenate(core_rec)
    numbers = {
        "core_ct": float(core_ct.mean()) if core_ct.size else np.nan,
        "core_reconstruction": float(core_rec.mean()) if core_rec.size else np.nan,
        "n_core_plane_voxels": int(core_ct.size),
    }
    return numbers, sums


def contrast_retention(
    core_ct: np.ndarray, core_rec: np.ndarray, grade: np.ndarray, g: int
) -> dict:
    """Contrast retention of grade `g` against grade 0 (definition in the module docstring).

    In: per-fruit core means of CT and reconstruction and the grade of each fruit.
    Out: the CT difference, the reconstruction difference and their ratio.
    """
    grade = np.asarray(grade)
    ct_difference = core_ct[grade == g].mean() - core_ct[grade == 0].mean()
    rec_difference = core_rec[grade == g].mean() - core_rec[grade == 0].mean()
    return {
        "ct_difference": float(ct_difference),
        "reconstruction_difference": float(rec_difference),
        "retention": float(rec_difference / ct_difference),
    }


# Bins of the latent height-radius map (Ruben, 27 October 2026: as fine as the grid allows).
# Height: the 2526 body spans 21 to 32 latent slices (median 28), so 20 bins give every fruit
# at least one slice per bin. Radius: the fruit is about 6 to 11 cells in radius (median about
# 7.5), so 8 bins over 0 to 1 are about one cell each for a median fruit.
N_LATENT_HEIGHT_BINS = 20
N_LATENT_RADIUS_BINS = 8
# The two ways to normalise the radius (notebook 07 compares them).
RADIUS_METHODS = ("fruit", "slice")


def latent_map_sums(
    latent: np.ndarray,
    body_cells: np.ndarray,
    radius_method: str = "fruit",
    n_height: int = N_LATENT_HEIGHT_BINS,
    n_radius: int = N_LATENT_RADIUS_BINS,
) -> tuple[np.ndarray, np.ndarray]:
    """Height-radius sums of one fruit's latent `mu`, over every body cell of the 3D grid.

    In: the latent (32³), the fruit's 32³ body cells and the radius method. Out: (counts,
    sums), each (n_height, n_radius): number of body cells per bin and the sum of their latent
    values. Unlike the voxel map, which only has the two saved long sections, this uses all
    body cells, so it averages over the full rotation about the axis.

    Height (both methods): 0 at the calyx end of the body, 1 at the stalk end.
    Radius, `radius_method`:
    - "fruit": distance to one straight stem-calyx axis (the line along the third axis
      through the mean x and y of all body cells), divided by one number per fruit, the
      distance of the outermost body cell. Keeps the shape: a narrow neck only reaches small
      radii.
    - "slice": distance to the centre of the body cells of the same slice, divided by the
      distance of the outermost body cell of that slice. Removes the shape: radius 1 is the
      skin at every height, so a radius bin is a relative depth below the skin.
    In both, radius runs from 0 to 1 and every body cell is binned. Neither method
    interpolates: every cell keeps its value and only gets a coordinate.
    """
    if radius_method not in RADIUS_METHODS:
        raise ValueError(f"radius_method must be one of {RADIUS_METHODS}")
    body = np.asarray(body_cells, dtype=bool)
    occupied = np.flatnonzero(body.any(axis=(0, 1)))
    low, high = int(occupied[0]), int(occupied[-1])
    gx, gy, gz = np.meshgrid(*(np.arange(n) for n in body.shape), indexing="ij")
    height = (gz - low) / (high - low + 1)

    if radius_method == "fruit":
        xs, ys, _ = np.nonzero(body)
        centre_x, centre_y = xs.mean(), ys.mean()
    else:
        # Centre per slice, broadcast along the third axis; empty slices get NaN (they hold
        # no body cells, so they never enter the bins).
        area = body.sum(axis=(0, 1))  # body cells per slice
        with np.errstate(invalid="ignore", divide="ignore"):
            centre_x = (gx * body).sum(axis=(0, 1)) / area
            centre_y = (gy * body).sum(axis=(0, 1)) / area
    distance = np.sqrt((gx - centre_x) ** 2 + (gy - centre_y) ** 2)
    body_distance = np.where(body, distance, 0.0)
    if radius_method == "fruit":
        outermost = body_distance.max()
    else:
        outermost = body_distance.max(axis=(0, 1))  # one value per slice
    # A slice of a single cell has outermost distance 0; its cell gets radius 0.
    with np.errstate(invalid="ignore", divide="ignore"):
        radius = np.where(outermost > 0, distance / outermost, 0.0)
    inside = body & (height >= 0) & (height < 1)
    h_bin = (height[inside] * n_height).astype(int)
    r_bin = radius_bin(radius[inside], n_radius)

    counts = np.zeros((n_height, n_radius))
    sums = np.zeros((n_height, n_radius))
    np.add.at(counts, (h_bin, r_bin), 1)
    np.add.at(sums, (h_bin, r_bin), np.asarray(latent, dtype=float)[inside])
    return counts, sums


# Group maps (notebooks 06 and 07, one rule everywhere; Ruben, 27 October 2026). A bin of a
# group is drawn only when at least this share of the group's fruit have cells in it: bins
# near the skin that only the widest fruit reach would describe those few fruit, not the group.
MIN_FRUIT_SHARE_PER_BIN = 0.5


def fruit_bin_means(counts: np.ndarray, sums: np.ndarray) -> np.ndarray:
    """Mean per bin of every fruit, NaN where the fruit has no cell or voxel in the bin.

    In: counts (fruit, height, radius) and sums (fruit, ..., height, radius) with any number
    of value axes between fruit and height (for example CT, reconstruction, residual).
    Out: means with the shape of `sums`.
    """
    counts = np.asarray(counts, dtype=float)
    sums = np.asarray(sums, dtype=float)
    # Broadcast the counts over the value axes.
    shaped = counts.reshape(
        counts.shape[:1] + (1,) * (sums.ndim - counts.ndim) + counts.shape[1:]
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(shaped > 0, sums / shaped, np.nan)


def group_bin_summary(
    means: np.ndarray,
    selected: np.ndarray,
    min_share: float = MIN_FRUIT_SHARE_PER_BIN,
) -> dict[str, np.ndarray]:
    """Mean, 25 % and 75 % per bin over the selected fruit, every fruit counting once.

    In: per-fruit bin means (fruit, ..., height, radius) from `fruit_bin_means`, a boolean
    selection over the fruit. Out: dict of "mean", "q25", "q75" (each ..., height, radius),
    NaN where fewer than `min_share` of the selected fruit have a value.
    """
    import warnings

    group = np.asarray(means)[np.asarray(selected, dtype=bool)]
    enough = (~np.isnan(group)).sum(axis=0) >= min_share * len(group)
    # Bins without any value give NaN on purpose; silence numpy's warning there.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        summary = {
            "mean": np.nanmean(group, axis=0),
            "q25": np.nanpercentile(group, 25, axis=0),
            "q75": np.nanpercentile(group, 75, axis=0),
        }
    return {k: np.where(enough, v, np.nan) for k, v in summary.items()}

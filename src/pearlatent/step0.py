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
  widest radius). Sums of CT, reconstruction and signed residual (reconstruction - CT) are
  collected in `N_HEIGHT_BINS` × `N_RADIUS_BINS` bins, so group means can be formed by adding
  sums and counts over fruit. Averaging over the two sections and over fruit removes the free
  rotation about the axis.
"""

from __future__ import annotations

import numpy as np

from pearlatent.pooling import FACTOR, MASK_THRESHOLD

# Bins of the height-radius map. Height 0 to 1 over the body. Radius 0 to 1.2 times the
# widest radius: a slice can be wider than the circle of the largest area when it is not round.
N_HEIGHT_BINS = 32
N_RADIUS_BINS = 24
MAX_RADIUS = 1.2


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
    `geometry` with plane_x, plane_y, axis_x, axis_y, widest_radius_voxels and
    core_radius_voxels (one row of `reconstruction.parquet`).
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
    for plane, index, axis_position in sections:
        ct = fruit_planes[0, plane]
        rec = fruit_planes[1, plane]
        body = body_on_plane(ct, body_cells, plane, index)
        across, along = np.meshgrid(np.arange(128), np.arange(128), indexing="ij")
        distance = np.abs(across - axis_position)
        core = body & (distance <= geometry["core_radius_voxels"])
        core_ct.append(ct[core])
        core_rec.append(rec[core])

        height = (along - low) / (high - low + 1)
        radius = distance / geometry["widest_radius_voxels"]
        inside = body & (height >= 0) & (height < 1) & (radius < MAX_RADIUS)
        h_bin = (height[inside] * N_HEIGHT_BINS).astype(int)
        r_bin = (radius[inside] / MAX_RADIUS * N_RADIUS_BINS).astype(int)
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

"""Shape, stalk and intensity numbers per fruit, measured on the 128³ volumes (23 October 2026).

Used as colours in the exploration plots and, later, as covariates in the orchard check
(charter rule 8). Every number is in voxels of the 128³ grid, so it is relative: each fruit was
rescaled to 128 voxels in length (with its stalk, in most 2526 volumes), so absolute size is not
in the volumes. If Hugo's rescaling was the same along all three axes, ratios such as
`diameter_to_length` are exact, and lengths and volumes follow in millimetres once the length of
each fruit before rescaling is known (open question 18).

Volume by voxel counting: threshold the scaled volume, keep the largest connected part, fill
enclosed holes, count voxels. The threshold is `MASK_THRESHOLD` (0.1 of the min-max scaled
intensity), the same as for the latent body masks, so the body here is the body the latent
features use.

Flat face (23 October 2026): a fruit whose volume misses a part has a flat face where the
field of view or the preprocessing cut it (`docs/outlier_rule.md`). Natural pear skin is curved
everywhere. `largest_flat_face` finds the plane that contains the most surface voxels of the
fruit (random sample consensus, RANSAC: planes through 3 random surface voxels, 3,000 planes,
seed 369, a voxel counts when it lies within 1 voxel of the plane) and keeps the largest
connected patch of those voxels. `flat_face_share` is that patch over all surface voxels.
Checked on 23 October on 22 fruit read by eye: cuts through the body 0.13 to 0.29, normal fruit
0.07 to 0.09 (the curved 2526_O19: 0.11). A cut off only at the tip of the neck is not
separated from normal fruit by this number (0.05 to 0.08) and is judged by eye.

Stalk present (23 October 2026): `stalk_length` of at least `STALK_PRESENT_VOXELS`. Shorter
values are the narrow tip of the neck, which `body_mask` also leaves out of the body (slices
with fewer than 4 latent cells): 38 of the 66 2425 validation fruit, which have no stalk, have
1 to 4 voxels, and only 1 of them has 10 or more.

Run for all open fruit of an encoding run (writes `derived/descriptors/<run_id>/`):

    uv run python -m pearlatent.descriptors
"""

from __future__ import annotations

import numpy as np

from pearlatent.pooling import MASK_THRESHOLD, body_mask, fruit_mask
from pearlatent.reconstruction import body_voxel_mask

ENCODING_RUN = "20261013T171306Z_hugo_vae"

# Flat-face search (module docstring): planes tried, distance to the plane in voxels, seed.
FLAT_FACE_PLANES = 3000
FLAT_FACE_TOLERANCE = 1.0
FLAT_FACE_SEED = 369
# Stalk counted as present from this many slices beyond the body (module docstring).
STALK_PRESENT_VOXELS = 10


def largest_part(solid: np.ndarray) -> np.ndarray:
    """The largest connected part of a boolean volume (6-connected), holes filled.

    Separates the fruit (with its stalk) from stray objects such as a speck of dust or a
    piece of the holder that lies above the threshold.
    """
    from scipy import ndimage

    labels, n = ndimage.label(solid)
    if n == 0:
        return np.zeros_like(solid, dtype=bool)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0  # label 0 is the background
    return ndimage.binary_fill_holes(labels == sizes.argmax())


def slice_extent(mask: np.ndarray) -> tuple[int, int]:
    """First and last index along the third axis (stem–calyx) that contain the mask; (-1, -1) if empty."""
    occupied = np.flatnonzero(mask.any(axis=(0, 1)))
    if occupied.size == 0:
        return -1, -1
    return int(occupied[0]), int(occupied[-1])


def flat_face_patch(
    solid: np.ndarray,
    n_planes: int = FLAT_FACE_PLANES,
    tolerance: float = FLAT_FACE_TOLERANCE,
    seed: int = FLAT_FACE_SEED,
) -> tuple[np.ndarray, int]:
    """The largest flat patch of the surface of `solid`, as a voxel mask, and the surface size.

    In: a boolean volume (the fruit). Out: (boolean mask of the patch, number of surface
    voxels). Surface voxels are voxels of `solid` with a neighbour outside it. Random sample
    consensus (RANSAC): a plane through 3 random surface voxels, the surface voxels within
    `tolerance` of it counted, repeated `n_planes` times (in batches of 200 for speed); the
    plane with the most voxels wins. The same seed gives the same planes.
    """
    from scipy import ndimage

    surface = solid & ~ndimage.binary_erosion(solid)
    points = np.argwhere(surface).astype(float)
    patch = np.zeros_like(surface, dtype=bool)
    if len(points) < 3:
        return patch, len(points)
    rng = np.random.default_rng(seed)
    best_count, best_plane = -1, None
    for _ in range(max(1, n_planes // 200)):
        index = rng.integers(0, len(points), size=(200, 3))
        a, b, c = points[index[:, 0]], points[index[:, 1]], points[index[:, 2]]
        normals = np.cross(b - a, c - a)
        lengths = np.linalg.norm(normals, axis=1)
        usable = lengths > 1e-6  # three points on a line define no plane
        normals = normals[usable] / lengths[usable, None]
        offsets = (normals * a[usable]).sum(axis=1)
        counts = (np.abs(points @ normals.T - offsets) <= tolerance).sum(axis=0)
        k = int(counts.argmax())
        if counts[k] > best_count:
            best_count, best_plane = int(counts[k]), (normals[k], offsets[k])
    normal, offset = best_plane
    near = np.abs(points @ normal - offset) <= tolerance
    # Only the largest connected patch counts: a plane through a curved surface also touches
    # it along a thin curve elsewhere.
    on_plane = np.zeros_like(surface, dtype=bool)
    on_plane[tuple(points[near].astype(int).T)] = True
    labels, n = ndimage.label(on_plane, structure=np.ones((3, 3, 3)))
    if n:
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0  # label 0 is everything off the plane
        patch = labels == sizes.argmax()
    return patch, len(points)


def largest_flat_face(
    solid: np.ndarray,
    n_planes: int = FLAT_FACE_PLANES,
    tolerance: float = FLAT_FACE_TOLERANCE,
    seed: int = FLAT_FACE_SEED,
) -> tuple[int, int]:
    """Size of the largest flat patch of the surface (`flat_face_patch`) and of the surface."""
    patch, surface = flat_face_patch(solid, n_planes, tolerance, seed)
    return int(patch.sum()), surface


def fruit_descriptors(scaled: np.ndarray) -> dict:
    """Numbers that describe one fruit.

    In: one min-max scaled 128³ volume, stem–calyx axis along the third axis, calyx at the low
    end. Out: a dict of numbers, all in voxels of the 128³ grid:

    - `fruit_voxels`: the fruit with stalk (largest part above the threshold, holes filled);
    - `stray_voxels`: voxels above the threshold outside that part (a stray object that may
      have entered the rescaling to 128 voxels);
    - `body_voxels`: the body without the stalk, as for the latent (`body_mask` at 32³, then
      fruit voxels inside it, holes filled; internal cavities count as body);
    - `body_length`: number of slices along the stem–calyx axis that contain body;
    - `stalk_length`: slices of fruit beyond the stalk end of the body (0 when no stalk);
    - `widest_diameter`: diameter of a circle with the area of the largest body slice;
    - `diameter_to_length`: `widest_diameter / body_length` (shape: round versus long);
    - `tissue_intensity_mean` and `tissue_intensity_median`: scaled intensity of the body
      voxels above the threshold, so cavities do not pull it down. Relative to the brightest
      voxel of the volume (min-max scaling), so a bright speck lowers it for the whole fruit.
    - `cavity_fraction`: share of the body below the threshold (enclosed air);
    - `stalk_present`: `stalk_length` ≥ `STALK_PRESENT_VOXELS`;
    - `flat_face_voxels`, `surface_voxels`, `flat_face_share`: the largest flat patch of the
      fruit surface (`largest_flat_face`), a sign that part of the fruit is cut off.
    """
    solid = np.asarray(scaled) > MASK_THRESHOLD
    fruit = largest_part(solid)
    body = body_voxel_mask(scaled, body_mask(fruit_mask(scaled)))
    tissue = body & solid

    fruit_low, fruit_high = slice_extent(fruit)
    body_low, body_high = slice_extent(body)
    body_length = body_high - body_low + 1 if body_low >= 0 else 0
    largest_slice_area = int(body.sum(axis=(0, 1)).max()) if body.any() else 0
    widest_diameter = 2 * float(np.sqrt(largest_slice_area / np.pi))
    body_count = int(body.sum())
    tissue_values = np.asarray(scaled)[tissue]
    flat_face, surface = largest_flat_face(fruit)
    stalk_length = int(max(fruit_high - body_high, 0)) if body_low >= 0 else 0

    return {
        "fruit_voxels": int(fruit.sum()),
        "stray_voxels": int((solid & ~fruit).sum()),
        "body_voxels": body_count,
        "body_length": int(body_length),
        # The stalk is at the high end of the third axis (calyx at the low end, notebook 02).
        "stalk_length": stalk_length,
        "stalk_present": stalk_length >= STALK_PRESENT_VOXELS,
        "widest_diameter": widest_diameter,
        "diameter_to_length": widest_diameter / body_length if body_length else np.nan,
        "tissue_intensity_mean": float(tissue_values.mean())
        if tissue_values.size
        else np.nan,
        "tissue_intensity_median": float(np.median(tissue_values))
        if tissue_values.size
        else np.nan,
        "cavity_fraction": float(1 - tissue.sum() / body_count)
        if body_count
        else np.nan,
        "flat_face_voxels": flat_face,
        "surface_voxels": surface,
        "flat_face_share": flat_face / surface if surface else np.nan,
    }


def main() -> None:
    """Measure every open fruit of `ENCODING_RUN` and write a parquet table with run.json."""
    import nibabel as nib
    import polars as pl

    from pearlatent.config import load_config
    from pearlatent.pooling import scale_minmax
    from pearlatent.runs import new_run_dir, write_run_record

    cfg = load_config()
    run = cfg.path("derived_dir") / "latents" / "hugo_vae" / ENCODING_RUN
    # Open fruit only: the sealed test fruit are not in this manifest and stay unopened.
    manifest = pl.read_parquet(run / "manifest.parquet")
    print(f"open fruit in {ENCODING_RUN}: {manifest.height}")

    rows = []
    for k, fruit in enumerate(manifest.iter_rows(named=True)):
        path = cfg.path(f"ct_dir_{fruit['season']}") / f"{fruit['file_stem']}.nii"
        scaled = scale_minmax(np.asarray(nib.load(path).dataobj))
        rows.append({"fruit_id": fruit["fruit_id"], **fruit_descriptors(scaled)})
        if (k + 1) % 100 == 0:
            print(f"  {k + 1} of {manifest.height}")

    table = pl.DataFrame(rows)
    out_dir = new_run_dir(cfg.path("derived_dir") / "descriptors", "descriptors")
    table.write_parquet(out_dir / "descriptors.parquet")
    write_run_record(
        out_dir,
        {
            "encoding_run": ENCODING_RUN,
            "fruit": table.height,
            "mask_threshold": MASK_THRESHOLD,
            "flat_face_planes": FLAT_FACE_PLANES,
            "flat_face_tolerance": FLAT_FACE_TOLERANCE,
            "flat_face_seed": FLAT_FACE_SEED,
            "stalk_present_voxels": STALK_PRESENT_VOXELS,
        },
    )
    print(f"wrote {table.height} fruit to {out_dir}")


if __name__ == "__main__":
    main()

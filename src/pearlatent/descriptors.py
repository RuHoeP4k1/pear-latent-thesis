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

Run for all open fruit of an encoding run (writes `derived/descriptors/<run_id>/`):

    uv run python -m pearlatent.descriptors
"""

from __future__ import annotations

import numpy as np

from pearlatent.pooling import MASK_THRESHOLD, body_mask, fruit_mask
from pearlatent.reconstruction import body_voxel_mask

ENCODING_RUN = "20261013T171306Z_hugo_vae"


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
    - `cavity_fraction`: share of the body below the threshold (enclosed air).
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

    return {
        "fruit_voxels": int(fruit.sum()),
        "stray_voxels": int((solid & ~fruit).sum()),
        "body_voxels": body_count,
        "body_length": int(body_length),
        # The stalk is at the high end of the third axis (calyx at the low end, notebook 02).
        "stalk_length": int(max(fruit_high - body_high, 0)) if body_low >= 0 else 0,
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
        },
    )
    print(f"wrote {table.height} fruit to {out_dir}")


if __name__ == "__main__":
    main()

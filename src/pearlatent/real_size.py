"""Real fruit size from Hugo's volumes before the rescaling to 128 voxels (27 November 2026).

Hugo's preprocessing (preprint section 2.1, MATLAB; Ruben, 27 November 2026: the same
procedure for 2526): scan voxels 0.6267 × 0.6267 × 0.2 mm, resampled to 0.6267 mm cubes by
linear interpolation, background removed, fruit separated, rotated so that the stem–calyx axis
is vertical, then rescaled to 128 voxels in length and zero-padded to 128³. On 27 November
Hugo gave the 2526 volumes of the step before the rescaling ("originals", config key
`ct_orig_dir_2526`): one cropped, aligned, segmented uint16 volume per fruit, stem–calyx axis
along the third array axis. Their header does not hold the voxel size (it says 1 mm); the
0.6267 mm comes from the preprint (`VOXEL_SIZE_MM`).

Checked on all 433 volumes on 27 November (scratch, then `rescaling_check` in this module):
- the original array is the tight crop of the fruit on every axis, so the length of the fruit
  is the length of the third array axis and can be read from the header alone;
- the rescaling is the same along all three axes: the factor along each short axis over the
  factor along the long axis is 0.993 ± 0.005 (range 0.980 to 1.012, one voxel of rounding);
- shrinking the original by `s` and centring it reproduces Hugo's 128³ volume (voxel
  correlation 0.985 to 0.992 on 2526_A01, C23, K07, O19).

So one number per fruit, the stretch factor s = 128 / original length in voxels (stalk
included), turns every length measured in the 128³ grid into millimetres: one 128³ voxel is
0.6267 / s mm wide. Example: an original length of 243 voxels (152 mm) gives s = 0.527 and
128³ voxels of 1.19 mm, so 150,000 body voxels are 150,000 × 1.19³ mm³ = 253 cm³.

Run (writes `derived/real_size/<run_id>/real_size.parquet` and `run.json`):

    uv run python -m pearlatent.real_size
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Edge of one original voxel after Hugo's resampling to cubes (preprint section 2.1).
# UNVERIFIED for 2526 from the files themselves (header says 1 mm); Ruben, 27 November 2026:
# the 2526 fruit followed the same procedure. A constant voxel size cancels in any comparison
# between fruit; it only sets the millimetre values.
VOXEL_SIZE_MM = 0.6267
# Length of every fruit after Hugo's rescaling, in voxels.
TARGET_LENGTH = 128
# The stem–calyx axis is the third array axis in the originals and the 128³ volumes.
LONG_AXIS = 2


def original_length(path: Path) -> int:
    """Length in voxels of the fruit along the stem–calyx axis, before the rescaling.

    In: path to one original volume. Out: the size of its third array axis. Only the header is
    read, not the voxels (the array is the tight crop of the fruit; module docstring).
    """
    import nibabel as nib

    return int(nib.load(path).shape[LONG_AXIS])


def stretch_factor(length: int | np.ndarray) -> float | np.ndarray:
    """s = 128 / original length: how much the fruit was shrunk (s < 1) or enlarged (s > 1)."""
    return TARGET_LENGTH / np.asarray(length, dtype=float)


def foreground_extent(mask: np.ndarray) -> tuple[int, int, int]:
    """Number of slices along each axis that contain at least one True voxel."""
    extents = []
    for axis in range(3):
        other_axes = tuple(a for a in range(3) if a != axis)
        filled = np.flatnonzero(mask.any(axis=other_axes))
        extents.append(int(filled[-1] - filled[0] + 1) if filled.size else 0)
    return tuple(extents)


def rescaling_check(original: np.ndarray, rescaled: np.ndarray) -> dict:
    """Was this fruit rescaled by the same factor along all three axes?

    In: the original volume (background is one constant floor value, its minimum) and Hugo's
    128³ volume (background 0). Out: the factor per axis (fruit extent after over before) and
    the factors of the two short axes divided by the factor of the long axis. Both ratios are
    1 for a uniform rescaling, up to one voxel of rounding (about 1 / 55 on the short axes).
    """
    before = foreground_extent(original > original.min())
    after = foreground_extent(rescaled > 0)
    factors = [after[a] / before[a] for a in range(3)]
    return {
        "factor_axis0": factors[0],
        "factor_axis1": factors[1],
        "factor_axis2": factors[2],
        "ratio_axis0_to_long": factors[0] / factors[LONG_AXIS],
        "ratio_axis1_to_long": factors[1] / factors[LONG_AXIS],
    }


def sizes_in_mm(
    body_voxels: np.ndarray,
    body_length: np.ndarray,
    stalk_length: np.ndarray,
    widest_diameter: np.ndarray,
    stretch: np.ndarray,
) -> dict[str, np.ndarray]:
    """Convert sizes measured in the 128³ grid (`pearlatent.descriptors`) to real units.

    In: per fruit, the body volume in 128³ voxels, the body length, stalk length and widest
    diameter in 128³ voxels, and the stretch factor. Out: body volume in cm³ and the three
    lengths in mm. One 128³ voxel is `VOXEL_SIZE_MM / s` mm wide (uniform rescaling).
    """
    voxel_mm = VOXEL_SIZE_MM / np.asarray(stretch, dtype=float)
    return {
        "voxel_mm_128": voxel_mm,
        "body_volume_cm3": np.asarray(body_voxels) * voxel_mm**3 / 1000.0,
        "body_length_mm": np.asarray(body_length) * voxel_mm,
        "stalk_length_mm": np.asarray(stalk_length) * voxel_mm,
        "widest_diameter_mm": np.asarray(widest_diameter) * voxel_mm,
    }


def main() -> None:
    """Stretch factor for every 2526 fruit; rescaling check and sizes for the open fruit."""
    import nibabel as nib
    import polars as pl

    from pearlatent.config import load_config
    from pearlatent.runs import (
        DESCRIPTORS_RUN_DIR,
        ENCODING_RUN_DIR,
        new_run_dir,
        write_run_record,
    )

    cfg = load_config()
    original_dir = cfg.path("ct_orig_dir_2526")
    rescaled_dir = cfg.path("ct_dir_2526")

    # Every 2526 fruit with a 128³ volume. Header only, so the sealed test fruit stay unopened.
    stems = sorted(p.stem for p in rescaled_dir.glob("*.nii"))
    missing = [s for s in stems if not (original_dir / f"{s}.nii").exists()]
    print(
        f"2526 fruit with a 128³ volume: {len(stems)}; without an original: {len(missing)}"
    )
    if missing:
        raise FileNotFoundError(f"no original for {missing}")
    lengths = [original_length(original_dir / f"{s}.nii") for s in stems]
    table = pl.DataFrame(
        {
            "fruit_id": [f"2526_{s}" for s in stems],
            "original_length_voxels": lengths,
        }
    ).with_columns(
        stretch_factor=TARGET_LENGTH / pl.col("original_length_voxels"),
        fruit_length_mm=pl.col("original_length_voxels") * VOXEL_SIZE_MM,
    )

    # Open fruit: those in the encoding manifest (the sealed fruit are not in it).
    manifest = pl.read_parquet(
        cfg.path("derived_dir") / ENCODING_RUN_DIR / "manifest.parquet"
    ).filter(pl.col("season") == "2526")
    open_ids = set(manifest["fruit_id"])
    print(f"open 2526 fruit (encoding manifest): {len(open_ids)}")

    checks = []
    for fruit_id in sorted(open_ids):
        stem = fruit_id.split("_", 1)[1]
        original = np.asarray(nib.load(original_dir / f"{stem}.nii").dataobj)
        rescaled = np.asarray(nib.load(rescaled_dir / f"{stem}.nii").dataobj)
        checks.append({"fruit_id": fruit_id, **rescaling_check(original, rescaled)})
    checks = pl.DataFrame(checks)

    descriptors = pl.read_parquet(
        cfg.path("derived_dir") / DESCRIPTORS_RUN_DIR / "descriptors.parquet"
    ).select(
        "fruit_id", "body_voxels", "body_length", "stalk_length", "widest_diameter"
    )
    open_table = table.join(descriptors, on="fruit_id", how="inner")
    print(f"open 2526 fruit with descriptors: {open_table.height}")
    mm = sizes_in_mm(
        open_table["body_voxels"].to_numpy(),
        open_table["body_length"].to_numpy(),
        open_table["stalk_length"].to_numpy(),
        open_table["widest_diameter"].to_numpy(),
        open_table["stretch_factor"].to_numpy(),
    )
    open_table = open_table.with_columns(
        [pl.Series(name, values) for name, values in mm.items()]
    ).select(["fruit_id", *mm.keys()])

    # One row per 2526 fruit; size and check columns are empty for the sealed fruit.
    result = table.join(open_table, on="fruit_id", how="left").join(
        checks, on="fruit_id", how="left"
    )
    print(
        "short-to-long axis factor ratio, open fruit: "
        f"axis 0 {checks['ratio_axis0_to_long'].mean():.3f} "
        f"± {checks['ratio_axis0_to_long'].std():.3f}, "
        f"axis 1 {checks['ratio_axis1_to_long'].mean():.3f} "
        f"± {checks['ratio_axis1_to_long'].std():.3f}"
    )

    out_dir = new_run_dir(cfg.path("derived_dir") / "real_size", "real_size")
    result.write_parquet(out_dir / "real_size.parquet")
    write_run_record(
        out_dir,
        {
            "fruit": result.height,
            "open_fruit_with_sizes": open_table.height,
            "voxel_size_mm": VOXEL_SIZE_MM,
            "target_length": TARGET_LENGTH,
            "descriptors_run": DESCRIPTORS_RUN_DIR,
            "encoding_run": ENCODING_RUN_DIR,
        },
    )
    print(f"written: {out_dir}")


if __name__ == "__main__":
    main()

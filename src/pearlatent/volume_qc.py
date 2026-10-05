"""Quality check of the CT volumes, computed from the voxel values without the encoder.

Facts seen on four volumes (5 October 2026, laptop): every file is 128³ uint16, the
background is exactly 0, intensities already span 0 to 65535, the NIfTI header gives a
voxel size of 1.0 on every axis (so it does not record the real voxel size), and the fruit
spans 126 to 128 voxels along the third array axis. `volume_stats` measures these facts on
every fruit instead of assuming them.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import polars as pl

EXPECTED_SHAPE = (128, 128, 128)
# Fraction of the maximum intensity (after min-max scaling to [0, 1]) above which a voxel
# counts as fruit. Background is exactly 0 and fruit tissue lies mostly above 0.7, so the
# mask is not sensitive to the exact value (checked at 0.05, 0.1 and 0.2 on four volumes).
DEFAULT_THRESHOLD = 0.1
# A fruit shorter than this along its longest axis is flagged: preprocessing should have
# rescaled every fruit to 128 voxels in length.
MIN_LONG_EXTENT = 120


def volume_stats(array: np.ndarray, threshold: float = DEFAULT_THRESHOLD) -> dict:
    """Measurements of one volume. `array` is the raw voxel array as stored in the file.

    The fruit mask is `scaled > threshold`, with `scaled` the volume min-max scaled to
    [0, 1] as in Hugo's loader. Extents are counted in voxels between the first and the last
    masked slice along each axis. `widest_slice_axis2` is the index along the third axis of
    the slice with the largest fruit cross-section; on a pear the wide end is the calyx end,
    so its position shows whether all fruit point the same way.
    """
    a = np.asarray(array)
    lo, hi = float(a.min()), float(a.max())
    stats = {
        "shape": "x".join(str(n) for n in a.shape),
        "dtype": str(a.dtype),
        "min": lo,
        "max": hi,
        "sha1": hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest(),
    }
    empty = {
        "fruit_voxels": 0,
        "extent_axis0": 0,
        "extent_axis1": 0,
        "extent_axis2": 0,
        "long_axis": None,
        "long_extent": 0,
        "widest_slice_axis2": None,
        "mean_in_fruit": None,
        "p05_in_fruit": None,
        "p95_in_fruit": None,
    }
    if a.ndim != 3 or hi <= lo:
        return stats | empty
    scaled = (a.astype(np.float32) - lo) / (hi - lo)
    mask = scaled > threshold
    n = int(mask.sum())
    if n == 0:
        return stats | empty
    extents = []
    for axis in range(3):
        present = np.flatnonzero(mask.any(axis=tuple(i for i in range(3) if i != axis)))
        extents.append(int(present[-1] - present[0] + 1))
    inside = scaled[mask]
    p05, p95 = np.percentile(inside, [5, 95])
    return stats | {
        "fruit_voxels": n,
        "extent_axis0": extents[0],
        "extent_axis1": extents[1],
        "extent_axis2": extents[2],
        "long_axis": int(np.argmax(extents)),
        "long_extent": max(extents),
        "widest_slice_axis2": int(np.argmax(mask.sum(axis=(0, 1)))),
        "mean_in_fruit": float(inside.mean()),
        "p05_in_fruit": float(p05),
        "p95_in_fruit": float(p95),
    }


def volume_header(path: str | Path) -> dict:
    """Voxel size and stored data type from the NIfTI header."""
    import nibabel as nib

    header = nib.load(str(path)).header
    zooms = [float(z) for z in header.get_zooms()[:3]]
    return {
        "header_dtype": str(header.get_data_dtype()),
        "voxel_size_0": zooms[0],
        "voxel_size_1": zooms[1],
        "voxel_size_2": zooms[2],
    }


def qc_table(
    volumes: pl.DataFrame, threshold: float = DEFAULT_THRESHOLD, progress=None
) -> pl.DataFrame:
    """One row per volume: the keys of `volumes` (output of `list_volumes`) plus the stats.

    `progress` optionally wraps the row iterator, for example `mo.status.progress_bar`.
    """
    import nibabel as nib

    rows = volumes.iter_rows(named=True)
    if progress is not None:
        rows = progress(list(rows))
    out = []
    for row in rows:
        img = nib.load(row["path"])
        out.append(
            {"fruit_id": row["fruit_id"]}
            | volume_header(row["path"])
            | volume_stats(np.asarray(img.dataobj), threshold)
        )
    return volumes.drop("path").join(
        pl.DataFrame(out, infer_schema_length=None), on="fruit_id", validate="1:1"
    )


def flag_volumes(
    qc: pl.DataFrame, min_long_extent: int = MIN_LONG_EXTENT
) -> pl.DataFrame:
    """Add a `qc_flag` column: empty string when the volume passes, else the reasons joined by "; "."""
    expected = "x".join(str(n) for n in EXPECTED_SHAPE)
    duplicated = pl.col("sha1").is_duplicated()
    reasons = [
        pl.when(pl.col("shape") != expected).then(pl.lit("shape not 128x128x128")),
        pl.when(pl.col("fruit_voxels") == 0).then(pl.lit("empty volume")),
        pl.when(
            (pl.col("fruit_voxels") > 0) & (pl.col("long_extent") < min_long_extent)
        ).then(pl.lit(f"long extent below {min_long_extent} voxels")),
        pl.when(duplicated).then(pl.lit("identical to another volume")),
    ]
    return qc.with_columns(
        pl.concat_str(reasons, separator="; ", ignore_nulls=True)
        .fill_null("")
        .alias("qc_flag")
    )

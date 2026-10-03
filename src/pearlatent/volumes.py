"""Listing CT volumes on disk. Only file names are read; no volume is opened."""

from __future__ import annotations

import re
from pathlib import Path

import polars as pl

from pearlatent.labels import NIFTI_SUFFIX, assert_unique_fruit, fruit_keys


def list_volumes(folder: str | Path, season: str) -> pl.DataFrame:
    """One row per NIfTI file in `folder` (not recursive), with the keys of `fruit_keys`.

    Columns: fruit_id, season, box, storage, file_stem, path. Sorted by file name.
    """
    files = sorted(
        p
        for p in Path(folder).iterdir()
        if p.is_file() and re.search(NIFTI_SUFFIX, p.name)
    )
    stems = pl.Series(
        [re.sub(NIFTI_SUFFIX, "", p.name) for p in files], dtype=pl.String
    )
    keys = fruit_keys(stems, season)
    assert_unique_fruit(keys["fruit_id"], str(folder))
    return keys.select(
        "fruit_id", "season", "box", "storage", "file_stem"
    ).with_columns(pl.Series("path", [str(p) for p in files], dtype=pl.String))

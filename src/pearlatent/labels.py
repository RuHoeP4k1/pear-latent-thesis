"""Label handling.

Tier 1 source: the two label CSV files and the grading rubric table. The expected
column names come from CLAUDE.md section 4; `read_label_file` refuses a file whose
header differs. The function `check_derived_labels` compares the derived columns
with any that already exist in the file, so a mismatch is caught instead of
silently replaced.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from pearlatent.config import Config

# Header of each label file (CLAUDE.md section 4)
EXPECTED_COLUMNS: dict[str, list[str]] = {
    "2425": ["filename", "rot", "browning", "cavity", "defective", "non-consumable"],
    "2526": ["filename", "browning", "cavity", "defective", "binary_123", "binary_23"],
}
# Integer grade columns; `non-consumable` is left out because its range is not known
GRADE_COLUMNS = ("rot", "browning", "cavity", "defective", "binary_123", "binary_23")
# Columns present in both seasons, in the order `load_labels` returns them
COMMON_COLUMNS = [
    "fruit_id",
    "season",
    "box",
    "storage",
    "file_stem",
    "browning",
    "cavity",
]

# File stem = box letter, optional "_opt", two-digit fruit number
NIFTI_SUFFIX = r"\.nii(\.gz)?$"
_NAME_PATTERNS: dict[str, str] = {
    "2425": r"^(?P<letter>[A-J])(?P<opt>_opt)?(?P<num>\d{2})$",
    "2526": r"^(?P<letter>[A-O])(?P<opt>_opt)?(?P<num>\d{2})$",
}
_letter = pl.col("letter")
_opt = pl.col("opt")
_num = pl.col("num")
_VALID: dict[str, pl.Expr] = {
    # 2425: A01-J60 per letter, plus G_opt01-30 and I_opt01-30
    "2425": pl.when(_opt.is_null())
    .then(_num.is_between(1, 60))
    .otherwise(_letter.is_in(["G", "I"]) & _num.is_between(1, 30)),
    # 2526: 15 boxes A-O of 30 fruit
    "2526": _opt.is_null() & _num.is_between(1, 30),
}
_STORAGE: dict[str, pl.Expr] = {
    # 2425 storage groups from CLAUDE.md section 4: 01-30 after suboptimal storage,
    # 31-60 at harvest, _opt after optimal storage
    "2425": pl.when(_opt.is_not_null())
    .then(pl.lit("optimal_storage"))
    .when(_num <= 30)
    .then(pl.lit("suboptimal_storage"))
    .otherwise(pl.lit("harvest")),
    # UNVERIFIED: the storage condition of the 2526 fruit is not known; all were
    # scanned after storage
    "2526": pl.lit("after_storage"),
}


def fruit_keys(stems: pl.Series, season: str) -> pl.DataFrame:
    """Build `file_stem, fruit_id, season, box, storage` from file stems such as `A01`.

    fruit_id is "<season>_<stem>" and box is "<season>_<letter>". The 2425
    optimal-storage fruit (G_opt, I_opt) join the box of their letter.
    # UNVERIFIED: G_opt and I_opt come from orchards G and I (decision of 3 October
    # 2026; to confirm with Hugo).
    Raises when a stem does not fit the naming scheme of its season.
    """
    if season not in _NAME_PATTERNS:
        raise ValueError(
            f"Unknown season {season!r}; expected one of {list(_NAME_PATTERNS)}"
        )
    parts = (
        stems.alias("file_stem")
        .to_frame()
        .with_columns(
            pl.col("file_stem").str.extract_groups(_NAME_PATTERNS[season]).alias("_p")
        )
        .unnest("_p")
        .with_columns(_num.cast(pl.Int32))
    )
    bad = parts.filter(~_VALID[season].fill_null(False))["file_stem"]
    if len(bad) > 0:
        raise ValueError(
            f"{len(bad)} file names of season {season} do not match the naming "
            f"scheme, e.g. {bad.head(3).to_list()}"
        )
    return parts.select(
        "file_stem",
        (pl.lit(season + "_") + pl.col("file_stem")).alias("fruit_id"),
        pl.lit(season).alias("season"),
        (pl.lit(season + "_") + _letter).alias("box"),
        _STORAGE[season].alias("storage"),
    )


def assert_unique_fruit(fruit_ids: pl.Series, what: str) -> None:
    dup = fruit_ids.filter(fruit_ids.is_duplicated()).unique().sort()
    if len(dup) > 0:
        raise ValueError(
            f"{len(dup)} fruit occur more than once in {what}, e.g. {dup.head(3).to_list()}"
        )


def read_label_file(path: str | Path, season: str) -> pl.DataFrame:
    """Read one label CSV file, add fruit keys and check header, duplicates and ranges.

    Returns the key columns of `fruit_keys` followed by every label column of the file.
    """
    header = pl.scan_csv(path).collect_schema().names()
    expected = EXPECTED_COLUMNS[season]
    if header != expected:
        missing = [c for c in expected if c not in header]
        extra = [c for c in header if c not in expected]
        raise ValueError(
            f"Header of {Path(path).name} differs from the expected columns: "
            f"missing {missing}, unexpected {extra}, order {header}"
        )
    raw = pl.scan_csv(
        path,
        schema_overrides={"filename": pl.String}
        | {c: pl.Int8 for c in GRADE_COLUMNS if c in header},
    ).collect()
    stems = raw["filename"].str.replace(NIFTI_SUFFIX, "")
    df = fruit_keys(stems, season).hstack(raw.drop("filename"))
    assert_unique_fruit(df["fruit_id"], Path(path).name)
    validate_ranges(df)
    return df


def load_labels(cfg: Config) -> pl.DataFrame:
    """Both label files, harmonised on the columns they share (`COMMON_COLUMNS`)."""
    df = pl.concat(
        [
            read_label_file(cfg.path(f"labels_{season}"), season).select(COMMON_COLUMNS)
            for season in EXPECTED_COLUMNS
        ]
    )
    assert_unique_fruit(df["fruit_id"], "the combined label table")
    return df


# Value ranges from the grading rubric (working rules, Tier 1)
LABEL_RANGES: dict[str, tuple[int, int]] = {
    "browning": (0, 3),
    "cavity": (0, 3),
    "rot": (0, 1),
}


def validate_ranges(
    df: pl.DataFrame, ranges: dict[str, tuple[int, int]] = LABEL_RANGES
) -> None:
    """Raise if a label column holds values outside its rubric range (nulls are reported separately)."""
    problems = []
    for col, (lo, hi) in ranges.items():
        if col not in df.columns:
            continue
        bad = df.filter(~pl.col(col).is_between(lo, hi) & pl.col(col).is_not_null())
        if len(bad) > 0:
            problems.append(f"{col}: {len(bad)} values outside [{lo}, {hi}]")
        n_null = df[col].null_count()
        if n_null:
            problems.append(f"{col}: {n_null} missing values")
    if problems:
        raise ValueError("Label check failed:\n  " + "\n  ".join(problems))


def derive_labels(df: pl.DataFrame) -> pl.DataFrame:
    """Add `defective_derived`, `binary_123_derived`, `binary_23_derived`.

    defective = max(browning, cavity); binary_123 = defective >= 1; binary_23 = defective >= 2.
    Confirmed with `check_derived_labels` on both label files (notebook 01, 3 October 2026):
    no disagreements.
    """
    defective = pl.max_horizontal("browning", "cavity")
    return df.with_columns(
        defective.alias("defective_derived"),
        (defective >= 1).cast(pl.Int8).alias("binary_123_derived"),
        (defective >= 2).cast(pl.Int8).alias("binary_23_derived"),
    )


def check_derived_labels(df: pl.DataFrame) -> pl.DataFrame:
    """Return a table of disagreements between derived and existing label columns (empty = agreement)."""
    pairs = [
        ("defective", "defective_derived"),
        ("binary_123", "binary_123_derived"),
        ("binary_23", "binary_23_derived"),
    ]
    present = [(a, b) for a, b in pairs if a in df.columns and b in df.columns]
    if not present:
        raise KeyError(
            "No existing label columns to compare with; run derive_labels first."
        )
    disagree = pl.any_horizontal(
        [pl.col(a).cast(pl.Int64) != pl.col(b).cast(pl.Int64) for a, b in present]
    )
    return df.filter(disagree)

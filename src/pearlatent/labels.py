"""Label handling.

Tier 1 source: the two label CSV files and the grading rubric table. The column
names below are placeholders until checked against the real CSV header; the
function `check_derived_labels` compares the derived columns with any that
already exist in the file, so a mismatch is caught instead of silently replaced.
"""

from __future__ import annotations

import polars as pl

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
    The thresholds are read from the names of the existing columns and must be
    confirmed with `check_derived_labels` against the CSV before use.
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

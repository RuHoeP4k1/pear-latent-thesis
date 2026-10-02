"""Train/test splits that respect the box structure of the data.

Rule (working rules, 2 October 2026): splits are always made by box, never by
fruit. All fruit of one box stay on the same side of every split, for every model
fitted on labels. The frozen encoder is not affected by this rule.

The box identifier is the letter in the file name. The exact file-name pattern is
not yet verified against the data, so the pattern is a required argument rather
than a default. Set it once in a notebook after inspecting real file names.
"""

from __future__ import annotations

import re

import numpy as np
import polars as pl


def box_from_filename(filenames: pl.Series, pattern: str) -> pl.Series:
    """Extract the box identifier with a regular expression that has one capture group.

    Fails loudly when a file name does not match, instead of producing nulls.
    """
    regex = re.compile(pattern)
    if regex.groups != 1:
        raise ValueError("pattern must contain exactly one capture group for the box")
    boxes = filenames.str.extract(pattern, group_index=1)
    missing = filenames.filter(boxes.is_null())
    if len(missing) > 0:
        raise ValueError(
            f"{len(missing)} file names do not match the box pattern, e.g. {missing.head(3).to_list()}"
        )
    return boxes.alias("box")


def group_kfold(
    df: pl.DataFrame,
    group_col: str = "box",
    n_splits: int = 5,
    stratify_col: str | None = None,
    seed: int = 0,
) -> pl.DataFrame:
    """Add a `fold` column (0 .. n_splits-1) assigned per group.

    With `stratify_col`, uses scikit-learn's StratifiedGroupKFold so each fold has a
    similar label distribution while groups stay intact.
    """
    from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

    groups = df[group_col].to_numpy()
    if stratify_col is None:
        splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = splitter.split(np.zeros(len(df)), groups=groups)
    else:
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=seed
        )
        split_iter = splitter.split(
            np.zeros(len(df)), df[stratify_col].to_numpy(), groups
        )

    fold = np.full(len(df), -1, dtype=np.int64)
    for k, (_, test_idx) in enumerate(split_iter):
        fold[test_idx] = k
    out = df.with_columns(pl.Series("fold", fold))
    assert_no_group_leakage(out, group_col=group_col, split_col="fold")
    return out


def assert_no_group_leakage(df: pl.DataFrame, group_col: str, split_col: str) -> None:
    """Raise if any group appears in more than one split. Call this before every fit."""
    leaking = (
        df.group_by(group_col)
        .agg(pl.col(split_col).n_unique().alias("n_splits"))
        .filter(pl.col("n_splits") > 1)
    )
    if len(leaking) > 0:
        raise AssertionError(
            f"Group leakage: {len(leaking)} {group_col} values occur in more than one "
            f"{split_col}: {leaking[group_col].head(5).to_list()}"
        )

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
from fractions import Fraction
from itertools import combinations

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


# ---------------------------------------------------------------------------------------------
# Test boxes and outer folds by restricted randomisation (docs/test_box_rule.md, approved by
# Ruben on 13 October 2026; charter rule 3). Section numbers below refer to that file.
# ---------------------------------------------------------------------------------------------

# The four balance counts of section 3, as conditions on one fruit.
BALANCE_COUNTS = {
    "non_consumable": pl.col("defective") >= 2,
    "browning_2plus_no_cavity": (pl.col("browning") >= 2) & (pl.col("cavity") == 0),
    "browning_3": pl.col("browning") == 3,
    "any_cavity": pl.col("cavity") >= 1,
}
# Tolerances tried in order (section 5.3 and 6.3); exact fractions, so no rounding.
TOLERANCES = (Fraction(30, 100), Fraction(40, 100), Fraction(50, 100))
# Fewer eligible sets or partitions than this widens the tolerance (sections 5.3 and 6.3).
MIN_ELIGIBLE = 10
SPLIT_SEED = 369


def box_counts(labels: pl.DataFrame) -> pl.DataFrame:
    """One row per box: `box, n_fruit` and the four balance counts.

    In: one row per scanned fruit with `box, browning, cavity, defective`.
    """
    return (
        labels.group_by("box")
        .agg(
            pl.len().alias("n_fruit"),
            *[cond.sum().alias(name) for name, cond in BALANCE_COUNTS.items()],
        )
        .sort("box")
    )


def set_passes(
    observed: int, total: int, n_set: int, n_reference: int, tolerance: Fraction
) -> bool:
    """Section 4: (1 - t) * E <= O <= (1 + t) * E with E = total * n_set / n_reference.

    Multiplied through by n_reference, so the comparison is between exact fractions:
    (1 - t) * total * n_set <= O * n_reference <= (1 + t) * total * n_set.
    A count exactly on a bound passes.
    """
    expected_times_n = Fraction(total * n_set)
    observed_times_n = Fraction(observed * n_reference)
    lower = (1 - tolerance) * expected_times_n
    upper = (1 + tolerance) * expected_times_n
    return lower <= observed_times_n <= upper


def _set_row(counts: pl.DataFrame, boxes: tuple[str, ...], totals: dict, n_ref: int):
    """Observed and expected counts of one set of boxes, and its eligibility per tolerance."""
    members = counts.filter(pl.col("box").is_in(boxes))
    n_set = int(members["n_fruit"].sum())
    row = {"boxes": "".join(b.split("_")[-1] for b in boxes), "n_fruit": n_set}
    for name in BALANCE_COUNTS:
        row[name] = int(members[name].sum())
        row[f"{name}_expected"] = totals[name] * n_set / n_ref
    for t in TOLERANCES:
        row[f"eligible_{float(t):.2f}"] = all(
            set_passes(row[name], totals[name], n_set, n_ref, t)
            for name in BALANCE_COUNTS
        )
    return row


def eligible_box_sets(counts: pl.DataFrame, k: int = 3) -> pl.DataFrame:
    """Section 5.1-5.2: all sets of k boxes in lexicographic order, with counts and eligibility.

    In: output of `box_counts` for the reference population (here the 433 scanned 2526 fruit).
    Out: one row per set (455 for 15 boxes and k = 3), with `eligible_0.30` etc.
    """
    totals = {name: int(counts[name].sum()) for name in BALANCE_COUNTS}
    n_ref = int(counts["n_fruit"].sum())
    boxes = sorted(counts["box"].to_list())
    rows = [_set_row(counts, s, totals, n_ref) for s in combinations(boxes, k)]
    return pl.DataFrame(rows)


def choose_tolerance(
    eligibility: dict[Fraction, int], minimum: int = MIN_ELIGIBLE
) -> Fraction:
    """The first tolerance with at least `minimum` eligible items; raise if none has.

    In: number of eligible sets (or partitions) per tolerance.
    """
    for t in TOLERANCES:
        if eligibility[t] >= minimum:
            return t
    raise RuntimeError(
        f"fewer than {minimum} eligible at every tolerance {eligibility}: stop and ask Ruben "
        "(docs/test_box_rule.md, sections 5.3 and 6.3)"
    )


def draw_test_boxes(
    sets: pl.DataFrame, tolerance: Fraction, seed: int = SPLIT_SEED
) -> dict:
    """Section 5.4: one eligible set, drawn with `default_rng(seed).integers(0, M)`."""
    eligible = sets.filter(pl.col(f"eligible_{float(tolerance):.2f}"))
    rng = np.random.default_rng(seed)
    k = int(rng.integers(0, eligible.height))
    return eligible.row(k, named=True)


def fold_partitions(boxes: list[str], fold_size: int = 3):
    """Section 6.1: every partition of `boxes` into folds of `fold_size`, in lexicographic order.

    The first remaining box always opens the next fold, and its partners run through the
    combinations of the other boxes in lexicographic order; so each partition is produced
    once, in the written order of the rule. 12 boxes in folds of 3 give 15,400 partitions.
    """
    remaining = sorted(boxes)
    if not remaining:
        yield ()
        return
    first, rest = remaining[0], remaining[1:]
    for partners in combinations(rest, fold_size - 1):
        fold = (first, *partners)
        others = [b for b in rest if b not in partners]
        for tail in fold_partitions(others, fold_size):
            yield (fold, *tail)


def eligible_partitions(counts: pl.DataFrame, fold_size: int = 3):
    """Section 6.2: all partitions of the boxes in `counts`, with eligibility per tolerance.

    In: output of `box_counts` for the 12 non-test boxes (the reference population of the
    folds, chosen by Ruben on 13 October 2026). A partition is eligible at t when each fold is.
    Out: list of partitions and a dict {tolerance: list of booleans}.
    """
    totals = {name: int(counts[name].sum()) for name in BALANCE_COUNTS}
    n_ref = int(counts["n_fruit"].sum())
    boxes = sorted(counts["box"].to_list())
    # Judge every possible fold once (220 for 12 boxes), then look partitions up.
    fold_ok = {
        fold: _set_row(counts, fold, totals, n_ref)
        for fold in combinations(boxes, fold_size)
    }
    partitions = list(fold_partitions(boxes, fold_size))
    eligible = {
        t: [
            all(fold_ok[fold][f"eligible_{float(t):.2f}"] for fold in p)
            for p in partitions
        ]
        for t in TOLERANCES
    }
    return partitions, eligible


def draw_partitions(
    partitions: list, eligible: list[bool], n_repeats: int = 5, seed: int = SPLIT_SEED
) -> list:
    """Section 6.4: `n_repeats` eligible partitions, `default_rng(seed).choice(P, n, replace=False)`."""
    candidates = [p for p, ok in zip(partitions, eligible, strict=True) if ok]
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(candidates), size=n_repeats, replace=False)
    return [candidates[int(i)] for i in picks]

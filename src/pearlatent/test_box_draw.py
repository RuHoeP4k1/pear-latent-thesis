"""Run the test-box draw and the outer-fold draw once (docs/test_box_rule.md, sections 5 to 7).

Run: `uv run python -m pearlatent.test_box_draw`. Writes into `derived/test_boxes/`:
`test_box_sets.csv` (all 455 sets), `test_boxes.csv` (the 3 drawn boxes), `test_box_fruit.csv`
(their fruit, read by `pearlatent.sealed`), `outer_folds.csv` (repeat, fold, box) and `run.json`.
Then rewrites `derived/sealed_fruit.csv`. Only labels are used.
"""

from __future__ import annotations

import subprocess
from collections import Counter
from itertools import combinations

import polars as pl

from pearlatent.exclusions import apply_exclusions, read_exclusions
from pearlatent.labels import read_label_file
from pearlatent.runs import write_run_record
from pearlatent.sealed import write_sealed_fruit
from pearlatent.splits import (
    BALANCE_COUNTS,
    SPLIT_SEED,
    TOLERANCES,
    box_counts,
    choose_tolerance,
    draw_partitions,
    draw_test_boxes,
    eligible_box_sets,
    eligible_partitions,
)

N_REPEATS = 5  # section 6.4
RULE_FILE = "docs/test_box_rule.md"


def _label(t) -> str:
    return f"{float(t):.2f}"


def rule_commit() -> str:
    """The last commit that changed the rule file; the draw must come after it."""
    out = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", RULE_FILE],
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


def scanned_2526_labels(cfg) -> pl.DataFrame:
    """Section 2: the 433 scanned 2526 fruit (the primary exclusions drop the 17 unscanned)."""
    labels = read_label_file(cfg.path("labels_2526"), "2526")
    exclusions = read_exclusions(cfg.path("derived_dir") / "exclusions.csv")
    scanned = apply_exclusions(labels, exclusions, "primary")
    print(f"2526 labels: {labels.height} fruit in, {scanned.height} scanned fruit out")
    if scanned.height != 433:
        raise ValueError(f"expected 433 scanned 2526 fruit, got {scanned.height}")
    return scanned


def draw_all(cfg) -> None:
    out_dir = cfg.path("derived_dir") / "test_boxes"
    out_dir.mkdir(parents=True, exist_ok=False)  # never overwrite an earlier draw
    commit = rule_commit()
    print(f"rule file {RULE_FILE} last committed in {commit[:7]}")

    labels = scanned_2526_labels(cfg)
    counts = box_counts(labels)
    print("season totals:", {n: int(counts[n].sum()) for n in BALANCE_COUNTS})

    # Section 5: the test boxes.
    sets = eligible_box_sets(counts)
    n_sets = {t: int(sets[f"eligible_{_label(t)}"].sum()) for t in TOLERANCES}
    print("eligible sets per tolerance:", {_label(t): n for t, n in n_sets.items()})
    t_test = choose_tolerance(n_sets)
    drawn = draw_test_boxes(sets, t_test, SPLIT_SEED)
    test_boxes = [f"2526_{letter}" for letter in drawn["boxes"]]
    print(f"tolerance used: {_label(t_test)}; drawn test boxes: {drawn['boxes']}")

    eligible_sets = sets.filter(pl.col(f"eligible_{_label(t_test)}"))
    box_frequency = Counter(letter for s in eligible_sets["boxes"] for letter in s)
    all_letters = [b.split("_")[1] for b in counts["box"]]
    print("occurrences of each box among the eligible sets:")
    print({letter: box_frequency.get(letter, 0) for letter in all_letters})

    # Section 6: the outer folds on the other 12 boxes.
    rest = counts.filter(~pl.col("box").is_in(test_boxes))
    print(f"boxes for the folds: {rest.height}, fruit: {int(rest['n_fruit'].sum())}")
    partitions, eligible = eligible_partitions(rest)
    n_parts = {t: sum(eligible[t]) for t in TOLERANCES}
    print(
        f"partitions: {len(partitions)}; eligible per tolerance:",
        {_label(t): n for t, n in n_parts.items()},
    )
    t_folds = choose_tolerance(n_parts)
    chosen = draw_partitions(partitions, eligible[t_folds], N_REPEATS, SPLIT_SEED)
    print(f"tolerance used for the folds: {_label(t_folds)}")

    fold_rows = []
    pair_counts = Counter()
    for repeat, partition in enumerate(chosen, start=1):
        written = " | ".join(
            "".join(b.split("_")[1] for b in fold) for fold in partition
        )
        print(f"  repeat {repeat}: {written}")
        for fold_number, fold in enumerate(partition, start=1):
            fold_rows += [
                {"repeat": repeat, "fold": fold_number, "box": b} for b in fold
            ]
            pair_counts.update(combinations(sorted(fold), 2))
    pairs = sorted(pair_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    print("pairs of boxes sharing a fold in more than one repeat:")
    for (a, b), n in pairs:
        if n > 1:
            print(f"  {a[-1]}{b[-1]}: {n} of {N_REPEATS}")

    # Section 7: write.
    sets.write_csv(out_dir / "test_box_sets.csv")
    pl.DataFrame([drawn]).write_csv(out_dir / "test_boxes.csv")
    test_fruit = labels.filter(pl.col("box").is_in(test_boxes)).select("fruit_id")
    test_fruit.write_csv(out_dir / "test_box_fruit.csv")
    pl.DataFrame(fold_rows).write_csv(out_dir / "outer_folds.csv")
    write_run_record(
        out_dir,
        {
            "what": "2526 test boxes and outer folds (docs/test_box_rule.md)",
            "rule_commit": commit,
            "seed": SPLIT_SEED,
            "tolerances_tried": [_label(t) for t in TOLERANCES],
            "eligible_sets": {_label(t): n for t, n in n_sets.items()},
            "tolerance_test_boxes": _label(t_test),
            "test_boxes": drawn["boxes"],
            "n_test_fruit": test_fruit.height,
            "eligible_partitions": {_label(t): n for t, n in n_parts.items()},
            "tolerance_folds": _label(t_folds),
            "n_repeats": N_REPEATS,
        },
    )
    print(f"written to {out_dir}")
    write_sealed_fruit(cfg)


if __name__ == "__main__":
    from pearlatent.config import load_config

    draw_all(load_config())

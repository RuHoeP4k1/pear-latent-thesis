"""Raise the outer-fold repeats from 5 to 20 (docs/test_box_rule.md, section 6.4, item 5).

Run: `uv run python -m pearlatent.extra_repeats`. Reads the draw of 13 October in
`derived/test_boxes/` (not changed), checks that its 5 partitions are reproduced, draws 15 more
from the eligible partitions not yet drawn and writes `derived/outer_folds_20/outer_folds.csv`
(repeat, fold, box; repeats 1 to 5 as before) and `run.json`. Only labels are used.
"""

from __future__ import annotations

import json
from collections import Counter
from itertools import combinations

import polars as pl

from pearlatent.config import load_config
from pearlatent.runs import write_run_record
from pearlatent.splits import (
    SPLIT_SEED,
    box_counts,
    draw_more_partitions,
    draw_partitions,
    eligible_partitions,
)
from pearlatent.test_box_draw import rule_commit, scanned_2526_labels

N_EARLIER = 5  # repeats drawn on 13 October
N_TOTAL = 20  # Ruben, 21 October 2026


def _written(partition) -> str:
    """A partition as box letters, for example 'ABI | CGN | DEO | HKL'."""
    return " | ".join("".join(b.split("_")[1] for b in fold) for fold in partition)


def earlier_partitions(folds: pl.DataFrame) -> list:
    """The partitions in `outer_folds.csv`, in repeat order, as tuples of sorted box tuples."""
    out = []
    for repeat in sorted(folds["repeat"].unique().to_list()):
        rows = folds.filter(pl.col("repeat") == repeat)
        partition = tuple(
            tuple(sorted(rows.filter(pl.col("fold") == f)["box"].to_list()))
            for f in sorted(rows["fold"].unique().to_list())
        )
        out.append(partition)
    return out


def main() -> None:
    cfg = load_config()
    draw_dir = cfg.path("derived_dir") / "test_boxes"
    out_dir = cfg.path("derived_dir") / "outer_folds_20"
    out_dir.mkdir(parents=True, exist_ok=False)  # never overwrite an earlier draw

    earlier_run = json.loads((draw_dir / "run.json").read_text())["params"]
    tolerance = earlier_run["tolerance_folds"]  # "0.30" on 13 October
    test_letters = earlier_run["test_boxes"]  # "FJM"
    test_boxes = [f"2526_{letter}" for letter in test_letters]

    labels = scanned_2526_labels(cfg)
    rest = box_counts(labels).filter(~pl.col("box").is_in(test_boxes))
    print(f"boxes for the folds: {rest.height}, fruit: {int(rest['n_fruit'].sum())}")
    partitions, eligible = eligible_partitions(rest)
    ok = next(v for t, v in eligible.items() if f"{float(t):.2f}" == tolerance)
    print(f"eligible partitions at ±{tolerance}: {sum(ok)} of {len(partitions)}")

    # Check: the 5 earlier repeats on disk equal a fresh run of the earlier draw.
    earlier = earlier_partitions(pl.read_csv(draw_dir / "outer_folds.csv"))
    if earlier != draw_partitions(partitions, ok, N_EARLIER, SPLIT_SEED):
        raise ValueError("outer_folds.csv does not match the draw of 13 October")
    print("earlier 5 repeats reproduced")

    more = draw_more_partitions(
        partitions, ok, earlier, N_TOTAL - N_EARLIER, SPLIT_SEED
    )
    chosen = earlier + more

    fold_rows = []
    pair_counts = Counter()
    for repeat, partition in enumerate(chosen, start=1):
        print(f"  repeat {repeat:2d}: {_written(partition)}")
        for fold_number, fold in enumerate(partition, start=1):
            fold_rows += [
                {"repeat": repeat, "fold": fold_number, "box": b} for b in fold
            ]
            pair_counts.update(combinations(sorted(fold), 2))
    n_pairs = len(list(combinations(sorted(rest["box"]), 2)))
    print(
        f"pairs of boxes that share a fold at least once: {len(pair_counts)} of {n_pairs}"
    )
    print("most frequent pairs:")
    for (a, b), n in pair_counts.most_common(5):
        print(f"  {a[-1]}{b[-1]}: {n} of {N_TOTAL}")

    pl.DataFrame(fold_rows).write_csv(out_dir / "outer_folds.csv")
    write_run_record(
        out_dir,
        {
            "what": "outer folds, 20 repeats (docs/test_box_rule.md section 6.4 item 5)",
            "rule_commit": rule_commit(),
            "seed": SPLIT_SEED,
            "tolerance_folds": tolerance,
            "test_boxes": test_letters,
            "eligible_partitions": sum(ok),
            "n_repeats": N_TOTAL,
            "repeats_from_13_october": N_EARLIER,
        },
    )


if __name__ == "__main__":
    main()

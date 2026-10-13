from fractions import Fraction
from math import comb

import polars as pl
import pytest

from pearlatent.splits import (
    TOLERANCES,
    box_counts,
    choose_tolerance,
    draw_partitions,
    draw_test_boxes,
    eligible_box_sets,
    eligible_partitions,
    fold_partitions,
    set_passes,
)

T30 = Fraction(30, 100)


def _fruit(box, n, browning=0, cavity=0):
    """n fruit of one box with the same grades; defective = max(browning, cavity)."""
    return [
        {
            "box": box,
            "browning": browning,
            "cavity": cavity,
            "defective": max(browning, cavity),
        }
    ] * n


@pytest.fixture
def labels():
    rows = []
    for i, box in enumerate("ABCDEF"):
        rows += _fruit(f"2526_{box}", 6)  # healthy
        rows += _fruit(
            f"2526_{box}", i % 3 + 1, browning=3
        )  # 1-3 browning 3, no cavity
        rows += _fruit(f"2526_{box}", 1, cavity=2)
    return pl.DataFrame(rows)


def test_box_counts(labels):
    counts = box_counts(labels)
    a = counts.filter(pl.col("box") == "2526_A").row(0, named=True)
    # 6 healthy + 1 browning 3 + 1 cavity 2
    assert a["n_fruit"] == 8
    assert a["non_consumable"] == 2
    assert a["browning_2plus_no_cavity"] == 1
    assert a["browning_3"] == 1
    assert a["any_cavity"] == 1


def test_example_of_the_rule():
    # E = 4.8 at t = 0.30: the window is 3.36 to 6.24 (docs/test_box_rule.md, section 4).
    # E = total * n_set / n_ref = 48 * 10 / 100.
    passing = [o for o in range(10) if set_passes(o, 48, 10, 100, T30)]
    assert passing == [4, 5, 6]


def test_exact_bound_passes_just_outside_fails():
    # E = 10: bounds 7 and 13 exactly.
    assert set_passes(7, 10, 1, 1, T30)
    assert set_passes(13, 10, 1, 1, T30)
    assert not set_passes(6, 10, 1, 1, T30)
    assert not set_passes(14, 10, 1, 1, T30)


def test_eligible_box_sets_counts_and_expected(labels):
    sets = eligible_box_sets(box_counts(labels))
    assert sets.height == comb(6, 3)
    assert sets["boxes"].to_list()[:2] == ["ABC", "ABD"]  # lexicographic
    abc = sets.filter(pl.col("boxes") == "ABC").row(0, named=True)
    # A, B, C have 1, 2, 3 browning-3 fruit: 6; set has 8 + 9 + 10 = 27 of 54 fruit
    assert abc["browning_3"] == 6
    assert abc["n_fruit"] == 27
    assert abc["browning_3_expected"] == pytest.approx(12 * 27 / 54)


def test_choose_tolerance_fallback():
    assert (
        choose_tolerance({TOLERANCES[0]: 12, TOLERANCES[1]: 30, TOLERANCES[2]: 50})
        == TOLERANCES[0]
    )
    assert (
        choose_tolerance({TOLERANCES[0]: 3, TOLERANCES[1]: 10, TOLERANCES[2]: 50})
        == TOLERANCES[1]
    )
    with pytest.raises(RuntimeError, match="ask Ruben"):
        choose_tolerance({TOLERANCES[0]: 0, TOLERANCES[1]: 1, TOLERANCES[2]: 9})


def test_draw_is_deterministic_and_eligible(labels):
    sets = eligible_box_sets(box_counts(labels))
    t = TOLERANCES[-1]
    first = draw_test_boxes(sets, t)
    assert first == draw_test_boxes(sets, t)
    assert first[f"eligible_{float(t):.2f}"]


def test_partition_count_and_each_box_once():
    boxes = list("ABCDEFGHIJKL")
    partitions = list(fold_partitions(boxes))
    assert len(partitions) == 15_400
    assert len(set(partitions)) == 15_400
    for p in partitions[::500]:
        flat = [b for fold in p for b in fold]
        assert sorted(flat) == boxes  # no box in two folds, none missing
    assert partitions[0] == (
        ("A", "B", "C"),
        ("D", "E", "F"),
        ("G", "H", "I"),
        ("J", "K", "L"),
    )
    assert partitions == sorted(partitions)  # lexicographic order of the written form


def test_draw_partitions_deterministic_distinct(labels):
    counts = box_counts(labels)
    partitions, eligible = eligible_partitions(counts)
    assert len(partitions) == 10  # 6 boxes in folds of 3
    ok = [True] * len(partitions)
    drawn = draw_partitions(partitions, ok, n_repeats=5)
    assert drawn == draw_partitions(partitions, ok, n_repeats=5)
    assert len(set(drawn)) == 5

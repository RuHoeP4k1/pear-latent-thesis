import polars as pl
import pytest

from pearlatent.labels import check_derived_labels, derive_labels, validate_ranges
from pearlatent.splits import assert_no_group_leakage, box_from_filename, group_kfold


@pytest.fixture
def toy() -> pl.DataFrame:
    # 8 boxes x 5 fruit; labels cycle so every fold sees several classes
    rows = []
    for b, box in enumerate("ABCDEFGH"):
        for i in range(5):
            rows.append(
                {
                    "file": f"pear_{box}_{i:03d}.mha",
                    "browning": (b + i) % 4,
                    "cavity": i % 2,
                    "rot": 0,
                }
            )
    return pl.DataFrame(rows)


def test_box_extraction(toy):
    boxes = box_from_filename(toy["file"], r"pear_([A-Z])_")
    assert boxes.n_unique() == 8


def test_box_extraction_fails_loudly(toy):
    with pytest.raises(ValueError, match="do not match"):
        box_from_filename(
            pl.Series(["pear_A_001.mha", "unexpected.mha"]), r"pear_([A-Z])_"
        )


@pytest.mark.parametrize("stratify", [None, "browning"])
def test_group_kfold_keeps_boxes_together(toy, stratify):
    df = toy.with_columns(box_from_filename(toy["file"], r"pear_([A-Z])_"))
    out = group_kfold(df, n_splits=4, stratify_col=stratify, seed=1)
    assert out["fold"].min() == 0 and out["fold"].max() == 3
    assert_no_group_leakage(out, "box", "fold")


def test_leakage_is_detected():
    df = pl.DataFrame({"box": ["A", "A", "B"], "fold": [0, 1, 1]})
    with pytest.raises(AssertionError, match="leakage"):
        assert_no_group_leakage(df, "box", "fold")


def test_derived_labels(toy):
    validate_ranges(toy)
    out = derive_labels(toy)
    expected = toy.select(pl.max_horizontal("browning", "cavity")).to_series()
    assert (out["defective_derived"] == expected).all()
    # Agreement with an "existing" column built the same way
    out = out.with_columns(pl.col("defective_derived").alias("defective"))
    assert check_derived_labels(out).is_empty()


def test_range_check_catches_bad_values():
    with pytest.raises(ValueError, match="outside"):
        validate_ranges(
            pl.DataFrame({"browning": [0, 4], "cavity": [0, 0], "rot": [0, 1]})
        )

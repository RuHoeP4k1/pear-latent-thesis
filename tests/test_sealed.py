import polars as pl
import pytest

from pearlatent.sealed import PART_2425, PART_2526, build_sealed_fruit, drop_sealed


@pytest.fixture
def hugo_split():
    return pl.DataFrame(
        {
            "fruit_id": ["2425_A01", "2425_A02", "2425_A03"],
            "hugo_split": ["train", "validation", "test"],
        }
    )


def test_only_hugo_test_fruit_sealed_before_the_draw(hugo_split):
    sealed = build_sealed_fruit(hugo_split)
    assert sealed.rows() == [("2425_A03", "2425", PART_2425)]


def test_test_boxes_added(hugo_split):
    boxes = pl.DataFrame({"fruit_id": ["2526_B01", "2526_B02"]})
    sealed = build_sealed_fruit(hugo_split, boxes)
    assert sealed.filter(pl.col("sealed_part") == PART_2526).height == 2
    assert sealed.height == 3


def test_duplicate_fruit_refused(hugo_split):
    with pytest.raises(ValueError, match="twice"):
        build_sealed_fruit(hugo_split, pl.DataFrame({"fruit_id": ["2425_A03"]}))


def test_drop_sealed_default_and_explicit(hugo_split):
    sealed = build_sealed_fruit(hugo_split)
    df = pl.DataFrame({"fruit_id": ["2425_A01", "2425_A02", "2425_A03"]})
    assert drop_sealed(df, sealed)["fruit_id"].to_list() == ["2425_A01", "2425_A02"]
    assert drop_sealed(df, sealed, include_sealed=True).height == 3

import polars as pl
import pytest

from pearlatent.fruit_table import CAUSES, attach_labels, cause, drop_faulty_volumes


def _tables():
    """A manifest of 4 fruit, their grades (one of each cause) and descriptors."""
    manifest = pl.DataFrame(
        {"fruit_id": ["2526_A01", "2526_A02", "2526_B01", "2526_B02"]}
    )
    labels = pl.DataFrame(
        {
            # Listed in another order than the manifest, with one extra fruit.
            "fruit_id": ["2526_B02", "2526_A01", "2526_B01", "2526_A02", "2526_C01"],
            "browning": [2, 0, 0, 3, 1],
            "cavity": [2, 0, 1, 0, 0],
        }
    )
    descriptors = pl.DataFrame(
        {
            "fruit_id": ["2526_A01", "2526_A02", "2526_B01", "2526_B02"],
            "stalk_length": [0, 12, 20, 3],
        }
    )
    return manifest, labels, descriptors


def test_cause_covers_the_four_cases():
    grades = pl.DataFrame({"browning": [0, 2, 0, 1], "cavity": [0, 0, 3, 1]})
    assert grades.select(cause())["cause"].to_list() == list(CAUSES)


def test_attach_labels_keeps_manifest_order_and_position():
    out = attach_labels(*_tables())
    assert out["fruit_id"].to_list() == ["2526_A01", "2526_A02", "2526_B01", "2526_B02"]
    assert out["position"].to_list() == [0, 1, 2, 3]
    assert out["defective"].to_list() == [0, 3, 1, 2]
    assert out["cause"].to_list() == ["none", "browning only", "cavity only", "both"]
    assert out["stalk_length"].to_list() == [0, 12, 20, 3]


def test_attach_labels_raises_on_a_fruit_without_grade():
    manifest, labels, descriptors = _tables()
    with pytest.raises(ValueError, match="without a grade"):
        attach_labels(
            manifest, labels.filter(pl.col("fruit_id") != "2526_B01"), descriptors
        )


def test_attach_labels_raises_on_a_fruit_without_descriptors():
    manifest, labels, descriptors = _tables()
    with pytest.raises(ValueError, match="without descriptors"):
        attach_labels(
            manifest, labels, descriptors.filter(pl.col("fruit_id") != "2526_A02")
        )


def test_drop_faulty_volumes_keeps_fruit_marked_keep(tmp_path):
    decisions = tmp_path / "volume_decisions.csv"
    pl.DataFrame(
        {
            "fruit_id": ["2526_A02", "2526_B01"],
            "season": ["2526", "2526"],
            "decision": ["missing_part", "keep"],
            "note": ["cut", ""],
        }
    ).write_csv(decisions)
    out = drop_faulty_volumes(attach_labels(*_tables()), decisions)
    assert out["fruit_id"].to_list() == ["2526_A01", "2526_B01", "2526_B02"]


def test_drop_faulty_volumes_without_file_changes_nothing(tmp_path):
    fruit = attach_labels(*_tables())
    assert drop_faulty_volumes(fruit, tmp_path / "absent.csv").equals(fruit)

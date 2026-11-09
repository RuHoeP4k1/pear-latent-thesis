import polars as pl
import pytest

from pearlatent.targets import TARGETS, target_table


@pytest.fixture
def fruit():
    # One fruit per combination that matters: (browning, cavity).
    return pl.DataFrame(
        {
            "fruit_id": ["a", "b", "c", "d", "e", "f"],
            "browning": [0, 1, 2, 3, 0, 3],
            "cavity": [0, 0, 0, 0, 2, 3],
            "defective": [0, 1, 2, 3, 2, 3],
        }
    )


def test_binary_123_keeps_all_fruit_and_cuts_at_grade_1(fruit):
    t = target_table(fruit, "binary_123")
    assert t["y"].to_list() == [0, 1, 1, 1, 1, 1]
    assert t["fit_on"].all()
    assert t["grade"].to_list() == [0, 1, 2, 3, 2, 3]


def test_cavity_free_target_drops_cavity_fruit_and_uses_browning(fruit):
    t = target_table(fruit, "browning_2plus_cavity_free")
    assert t["fruit_id"].to_list() == ["a", "b", "c", "d"]
    assert t["y"].to_list() == [0, 0, 1, 1]
    assert t["fit_on"].all()


def test_severity_fits_on_grades_0_and_3_only_but_keeps_all_fruit(fruit):
    t = target_table(fruit, "severity_defective")
    assert t["fruit_id"].to_list() == ["a", "b", "c", "d", "e", "f"]
    assert t["fit_on"].to_list() == [True, False, False, True, False, True]
    assert t["y"].to_list() == [0, 0, 0, 1, 0, 1]


def test_severity_on_browning_is_cavity_free(fruit):
    t = target_table(fruit, "severity_browning_cavity_free")
    assert t["fruit_id"].to_list() == ["a", "b", "c", "d"]
    assert t["fit_on"].to_list() == [True, False, False, True]


def test_unknown_target_is_refused(fruit):
    with pytest.raises(ValueError):
        target_table(fruit, "nonsense")
    assert "binary_123" in TARGETS

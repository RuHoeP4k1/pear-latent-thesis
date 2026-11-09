import numpy as np
import polars as pl

from pearlatent.counterfactual import (
    draw_example_fruit,
    plsda_direction,
    shift_to_score,
)


def test_shift_reaches_the_target_score_with_the_smallest_change():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (60, 30))
    y = (x[:, 0] + 0.3 * rng.normal(0, 1, 60) > 0).astype(float)
    direction = plsda_direction(x, y, n_components=3)
    row = x[5]
    moved = shift_to_score(row, direction, target=0.9)
    assert np.isclose(direction.predict(moved), 0.9)
    # The change lies along b: the smallest change that gives this score.
    change = moved - row
    b = direction.coefficients
    assert np.isclose(abs(change @ b), np.linalg.norm(change) * np.linalg.norm(b))


def _fruit():
    rows = []
    for box in "ABCDE":
        for k in range(1, 7):
            browning = 3 if k <= 2 else 0
            cavity = 1 if k == 2 else 0  # one grade 3 fruit per box has a cavity
            rows.append(
                {
                    "fruit_id": f"2526_{box}{k:02d}",
                    "box": f"2526_{box}",
                    "browning": browning,
                    "cavity": cavity,
                    "defective": max(browning, cavity),
                }
            )
    return pl.DataFrame(rows)


def test_four_fruit_from_four_boxes_cavity_free():
    drawn = draw_example_fruit(_fruit(), seed=369)
    assert drawn.height == 4
    assert drawn["box"].n_unique() == 4
    assert (drawn["cavity"] == 0).all()
    assert drawn.filter(pl.col("group") == "grade 3")["browning"].to_list() == [3, 3]
    assert drawn.filter(pl.col("group") == "grade 0")["defective"].to_list() == [0, 0]


def test_excluded_fruit_is_replaced_and_the_rest_stays():
    fruit = _fruit()
    first = draw_example_fruit(fruit, seed=369)
    rejected = first["fruit_id"][1]
    second = draw_example_fruit(fruit, seed=369, exclude=(rejected,))
    assert rejected not in second["fruit_id"].to_list()
    assert second["fruit_id"][0] == first["fruit_id"][0]
    assert second.height == 4 and second["box"].n_unique() == 4


def test_same_seed_same_fruit_whatever_the_row_order():
    fruit = _fruit()
    a = draw_example_fruit(fruit, seed=369)["fruit_id"].to_list()
    b = draw_example_fruit(fruit.reverse(), seed=369)["fruit_id"].to_list()
    assert a == b

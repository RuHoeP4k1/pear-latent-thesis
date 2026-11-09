import polars as pl

from pearlatent.counterfactual import draw_example_fruit


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


def test_same_seed_same_fruit_whatever_the_row_order():
    fruit = _fruit()
    a = draw_example_fruit(fruit, seed=369)["fruit_id"].to_list()
    b = draw_example_fruit(fruit.reverse(), seed=369)["fruit_id"].to_list()
    assert a == b

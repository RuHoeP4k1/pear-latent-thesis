import numpy as np
import pytest

from pearlatent.component_choice import (
    box_sums,
    choose_n_components,
    sign_flip_p_value,
)


def test_box_sums():
    assert box_sums([1, 2, 3, 4], ["b", "a", "b", "a"]).tolist() == [6, 4]


def test_sign_flip_all_positive_gives_smallest_p():
    # 12 boxes all worse: only the observed combination reaches the observed sum.
    t, p = sign_flip_p_value(np.ones(12))
    assert t == 12
    assert p == pytest.approx(1 / 4096)


def test_sign_flip_symmetric_gives_large_p():
    # Half the boxes better, half worse by the same amount: T = 0, about half the
    # combinations reach it.
    t, p = sign_flip_p_value(np.array([1.0, -1.0] * 6))
    assert t == 0
    assert p > 0.5


def test_choose_picks_fewest_not_worse():
    rng = np.random.default_rng(0)
    boxes = np.repeat(np.arange(12), 20)
    base = rng.uniform(0.1, 0.2, size=boxes.size)
    # h = 1 clearly worse in every box; h = 2 and h = 3 equal up to noise; h = 3 the minimum.
    errors = np.column_stack(
        [base + 0.2, base + rng.normal(0, 0.01, boxes.size) + 0.001, base]
    )
    result = choose_n_components(errors, boxes)
    assert result["h_star"] == 3
    assert result["tests"][1]["p"] < 0.10
    assert result["chosen"] == 2


def test_choose_keeps_h_star_when_all_smaller_are_worse():
    boxes = np.repeat(np.arange(12), 5)
    errors = np.column_stack([np.full(60, 0.5), np.full(60, 0.3), np.full(60, 0.1)])
    result = choose_n_components(errors, boxes)
    assert result["h_star"] == 3 and result["chosen"] == 3

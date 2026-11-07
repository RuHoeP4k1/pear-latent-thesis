import numpy as np
import polars as pl
import pytest

from pearlatent.ordinal import (
    cross_validated_ordinal,
    fit_ordinal,
    slope_and_cutpoints,
)

BOXES = [f"B{k:02d}" for k in range(12)]


def _data(seed: int = 0):
    """12 boxes of 25 fruit; the score rises with the grade, with noise."""
    rng = np.random.default_rng(seed)
    boxes = np.repeat(BOXES, 25)
    grade = rng.integers(0, 4, boxes.size)
    score = 0.3 * grade + rng.normal(0, 0.15, boxes.size)
    return score, grade, boxes


def _folds(n_repeats: int = 2) -> pl.DataFrame:
    rng = np.random.default_rng(1)
    rows = []
    for r in range(1, n_repeats + 1):
        order = rng.permutation(BOXES)
        for f in range(4):
            rows += [
                {"repeat": r, "fold": f + 1, "box": b} for b in order[3 * f : 3 * f + 3]
            ]
    return pl.DataFrame(rows)


def test_cutpoints_lie_between_the_grade_means():
    score, grade, _ = _data()
    slope, cutpoints = slope_and_cutpoints(fit_ordinal(score, grade, (0, 1, 2, 3)))
    assert slope > 0
    boundaries = cutpoints / slope  # on the score scale
    means = [score[grade == k].mean() for k in range(4)]
    for j in range(3):
        assert means[j] < boundaries[j] < means[j + 1]


def test_probabilities_sum_to_one_and_grades_are_recovered():
    score, grade, boxes = _data()
    probabilities = cross_validated_ordinal(score, grade, boxes, _folds())
    assert probabilities.shape == (grade.size, 4)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    assert (probabilities.argmax(axis=1) == grade).mean() > 0.7


def test_missing_class_is_refused():
    score, grade, _ = _data()
    keep = grade != 2
    with pytest.raises(ValueError, match="without fruit"):
        fit_ordinal(score[keep], grade[keep], (0, 1, 2, 3))

import numpy as np
import polars as pl
import pytest
from sklearn.metrics import roc_auc_score

from pearlatent.plsda import cross_validated_predictions
from pearlatent.svm import cross_validated_decisions, settings_grid

BOXES = [f"B{k:02d}" for k in range(12)]


def _folds(n_repeats: int = 2, seed: int = 1) -> pl.DataFrame:
    """Random partitions of the 12 boxes into 4 folds of 3."""
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(1, n_repeats + 1):
        order = rng.permutation(BOXES)
        for f in range(4):
            rows += [
                {"repeat": r, "fold": f + 1, "box": b} for b in order[3 * f : 3 * f + 3]
            ]
    return pl.DataFrame(rows)


def _spread_data(seed: int = 0):
    """12 boxes of 20 fruit, 20 features. Class 1 has the same mean as class 0 in every
    feature but twice the spread in features 0 to 4: invisible to a linear model, visible to
    a distance-based one (the situation of the interior SD, 20 November 2026)."""
    rng = np.random.default_rng(seed)
    boxes = np.repeat(BOXES, 20)
    y = rng.integers(0, 2, boxes.size)
    x = rng.normal(size=(boxes.size, 20))
    x[:, :5] *= np.where(y == 1, 2.0, 1.0)[:, None]
    return x, y, boxes


def test_grid_is_twelve_pairs_with_c_slowest():
    grid = settings_grid()
    assert len(grid) == 12
    assert grid[0] == (0.1, 0.1) and grid[1] == (0.1, 1.0) and grid[3] == (1.0, 0.1)


def test_every_fruit_scored_once_per_repeat_and_spread_found():
    x, y, boxes = _spread_data()
    decisions = cross_validated_decisions(x, y, boxes, _folds(), [(1.0, 1.0)])
    assert decisions.shape == (2, len(y), 1)
    assert roc_auc_score(y, decisions.mean(axis=0)[:, 0]) > 0.7
    # The same data through linear PLS-DA: near chance, because the class means are equal.
    pls = cross_validated_predictions(
        x, y, boxes, _folds(), scale=False, max_components=3
    )
    assert abs(roc_auc_score(y, pls.mean(axis=0)[:, 2]) - 0.5) < 0.12


def test_no_signal_gives_chance_auc():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(240, 20))
    y = rng.integers(0, 2, 240)
    boxes = np.repeat(BOXES, 20)
    decisions = cross_validated_decisions(x, y, boxes, _folds(), [(1.0, 1.0)])
    assert abs(roc_auc_score(y, decisions.mean(axis=0)[:, 0]) - 0.5) < 0.12


def test_box_missing_from_folds_raises():
    x, y, boxes = _spread_data()
    folds = _folds().filter(pl.col("box") != "B00")
    with pytest.raises(ValueError, match="without a fold"):
        cross_validated_decisions(x, y, boxes, folds, [(1.0, 1.0)])

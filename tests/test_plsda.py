import numpy as np
import polars as pl
import pytest
from sklearn.cross_decomposition import PLSRegression

from pearlatent.component_choice import choose_n_components
from pearlatent.plsda import (
    box_bootstrap_auc,
    classification_scores,
    cross_validated_predictions,
    mean_squared_errors,
    predict_each_h,
)

BOXES = [f"B{k:02d}" for k in range(12)]


def _data(signal: float, seed: int = 0):
    """12 boxes of 20 fruit, 300 features; the label shifts features 0 to 9 by `signal`."""
    rng = np.random.default_rng(seed)
    boxes = np.repeat(BOXES, 20)
    y = rng.integers(0, 2, boxes.size)
    x = rng.normal(size=(boxes.size, 300))
    x[:, :10] += signal * y[:, None]
    # A box effect on other features, as the orchards have.
    box_shift = rng.normal(size=(12, 300))
    x += box_shift[np.repeat(np.arange(12), 20)] * 0.5
    return x, y, boxes


def _folds(n_repeats: int = 3, seed: int = 1) -> pl.DataFrame:
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


@pytest.mark.parametrize("scale", [True, False])
def test_predictions_for_each_h_equal_refitting(scale):
    x, y, _ = _data(signal=1.0)
    model = PLSRegression(n_components=6, scale=scale).fit(
        x[:150], y[:150].astype(float)
    )
    every_h = predict_each_h(model, x[150:])
    for h in (1, 3, 6):
        refit = PLSRegression(n_components=h, scale=scale).fit(
            x[:150], y[:150].astype(float)
        )
        np.testing.assert_allclose(
            every_h[:, h - 1], refit.predict(x[150:]).ravel(), atol=1e-8
        )


def test_every_fruit_predicted_once_per_repeat_and_signal_found():
    x, y, boxes = _data(signal=1.0)
    predictions = cross_validated_predictions(
        x, y, boxes, _folds(), scale=False, max_components=5
    )
    assert predictions.shape == (3, len(y), 5)
    assert np.isfinite(predictions).all()
    scores = classification_scores(predictions.mean(axis=0)[:, 2], y, y)
    assert scores["auc"] > 0.9


def test_shuffled_labels_give_chance_auc():
    x, y, boxes = _data(signal=0.0)
    predictions = cross_validated_predictions(
        x, y, boxes, _folds(), scale=False, max_components=3
    )
    auc = classification_scores(predictions.mean(axis=0)[:, 0], y, y)["auc"]
    assert 0.35 < auc < 0.65


def test_box_in_two_folds_is_refused():
    x, y, boxes = _data(signal=1.0)
    folds = _folds(n_repeats=1)
    broken = pl.concat(
        [folds, pl.DataFrame({"repeat": [1], "fold": [2], "box": ["B00"]})]
    )
    with pytest.raises(ValueError, match="more than one fold"):
        cross_validated_predictions(x, y, boxes, broken, scale=False, max_components=2)


def test_box_without_fold_is_refused():
    x, y, boxes = _data(signal=1.0)
    folds = _folds(n_repeats=1).filter(pl.col("box") != "B05")
    with pytest.raises(ValueError, match="without a fold"):
        cross_validated_predictions(x, y, boxes, folds, scale=False, max_components=2)


def test_press_curve_and_component_choice_run_together():
    x, y, boxes = _data(signal=1.0)
    predictions = cross_validated_predictions(
        x, y, boxes, _folds(), scale=False, max_components=6
    )
    errors = mean_squared_errors(predictions, y)
    assert errors.shape == (len(y), 6)
    # Column sums of the averaged squared errors equal the mean PRESS over repeats.
    press_per_repeat = ((predictions - y[None, :, None]) ** 2).sum(axis=1)
    np.testing.assert_allclose(errors.sum(axis=0), press_per_repeat.mean(axis=0))
    choice = choose_n_components(errors, boxes)
    assert 1 <= choice["chosen"] <= choice["h_star"] <= 6


def test_called_positive_per_grade_and_bootstrap():
    prediction = np.array([0.9, 0.8, 0.2, 0.1, 0.6, 0.4])
    y = np.array([1, 1, 0, 0, 1, 0])
    grade = np.array([3, 2, 0, 0, 2, 1])
    scores = classification_scores(prediction, y, grade)
    assert scores["called_positive_grade_2"] == 1.0
    assert scores["called_positive_grade_1"] == 0.0
    assert scores["precision"] == 1.0
    x, y2, boxes = _data(signal=1.0)
    p = x[:, :10].mean(axis=1)
    low, high, aucs = box_bootstrap_auc(p, y2, boxes, n_resamples=200)
    assert low <= np.median(aucs) <= high

"""PLS-DA under design A (`docs/cv_explained.md`, sections 4 to 7).

Design A, in the order the functions are used:

1. `cross_validated_predictions`: for every repeat of the box cross-validation (the 20
   partitions of `derived/outer_folds_20/outer_folds.csv`) and every fold, fit PLS on the fruit
   of the other 9 boxes and predict the 3 held-out boxes with 1 to `max_components` latent
   variables. Every fruit is predicted once per repeat.
2. `mean_squared_errors`: per fruit and number of latent variables, the squared error averaged
   over the repeats. Its column sums are the mean PRESS curve.
3. `pearlatent.component_choice.choose_n_components`: the number of latent variables (van der
   Voet randomisation test, signs flipped per box).
4. `classification_scores` and `box_bootstrap_auc`: area under the ROC curve (AUC), recall per
   grade and precision at the fixed cut-off 0.5, and the 95 % interval by a bootstrap over boxes.

PLS-DA here is PLS regression on the 0/1 label (scikit-learn `PLSRegression`, NIPALS). For one
response, the model with h latent variables is the first h latent variables of the model with
more: the components are found one after the other, and the rotation matrix W (PᵀW)⁻¹ has the
property that its first h columns equal the rotation of the h-component model (PᵀW is upper
triangular). So one fit with `max_components` gives the predictions for every h
(`predict_each_h`, checked against refitting in the tests).

Whether the cells are scaled to unit variance (`scale`) has no default on purpose: it is a
choice of the analysis plan (CLAUDE.md section 7 says "standardise inside the training fold";
notebook 05 did not scale for the PCA because background cells hardly vary). Scaling, when
used, is computed by `PLSRegression` from the training fruit only (charter rule 9).
"""

from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import roc_auc_score

from pearlatent.splits import assert_no_group_leakage

# Fixed in advance (`docs/cv_explained.md`, section 4).
MAX_COMPONENTS = 15
CUTOFF = 0.5
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 369


def predict_each_h(model: PLSRegression, x: np.ndarray) -> np.ndarray:
    """Predictions of a fitted one-response PLS model with 1, 2, ..., H latent variables.

    In: the fitted model (H components) and new rows `x`. Out: an array (rows, H); column h-1
    holds the prediction with h latent variables.
    """
    # `_x_mean`, `_x_std`, `_y_mean`, `_y_std` are scikit-learn's stored training means and
    # scales (private names); the test against refitting fails if a version renames them.
    x_centred = (x - model._x_mean) / model._x_std
    scores = x_centred @ model.x_rotations_  # (rows, H)
    # Contribution of each latent variable to the prediction, in the units of y.
    contributions = scores * model.y_loadings_[0] * model._y_std[0]
    return model._y_mean[0] + np.cumsum(contributions, axis=1)


def cross_validated_predictions(
    x: np.ndarray,
    y: np.ndarray,
    boxes: np.ndarray,
    folds: pl.DataFrame,
    scale: bool,
    max_components: int = MAX_COMPONENTS,
) -> np.ndarray:
    """Out-of-fold predictions for every repeat, fruit and number of latent variables.

    In: features `x` (fruit × features), 0/1 labels `y`, the box of every fruit, the fold table
    (columns repeat, fold, box) and the scaling choice. Out: an array (repeats, fruit,
    max_components). Raises if a fruit's box is missing from a repeat, if a box sits in two folds
    of one repeat, or if a box is on both sides of a fit.
    """
    boxes = np.asarray(boxes)
    repeats = sorted(folds["repeat"].unique().to_list())
    out = np.full((len(repeats), len(y), max_components), np.nan)
    for r_index, repeat in enumerate(repeats):
        partition = folds.filter(pl.col("repeat") == repeat)
        if partition["box"].n_unique() != partition.height:
            raise ValueError(f"repeat {repeat}: a box is in more than one fold")
        fold_of_box = dict(zip(partition["box"], partition["fold"], strict=True))
        missing = set(boxes) - set(fold_of_box)
        if missing:
            raise ValueError(
                f"repeat {repeat}: boxes without a fold: {sorted(missing)}"
            )
        fold_of_fruit = np.array([fold_of_box[b] for b in boxes])
        for fold in sorted(set(fold_of_fruit)):
            held_out = fold_of_fruit == fold
            sides = pl.DataFrame(
                {"box": boxes, "side": np.where(held_out, "test", "train")}
            )
            assert_no_group_leakage(sides, "box", "side")
            model = PLSRegression(n_components=max_components, scale=scale)
            model.fit(x[~held_out], y[~held_out].astype(float))
            out[r_index, held_out] = predict_each_h(model, x[held_out])
    if np.isnan(out).any():
        raise AssertionError("some fruit were not predicted in some repeat")
    return out


def mean_squared_errors(predictions: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Squared error per repeat, averaged over the repeats: an array (fruit, H).

    Its column sums are the mean PRESS curve (`docs/cv_explained.md`, section 4, items a, b).
    """
    return ((predictions - np.asarray(y, dtype=float)[None, :, None]) ** 2).mean(axis=0)


def classification_scores(
    prediction: np.ndarray, y: np.ndarray, grade: np.ndarray, cutoff: float = CUTOFF
) -> dict:
    """AUC, and at the fixed cut-off: precision and the share called positive per grade.

    For grades with label 1 the share called positive is the recall; for grades with label 0
    it is the false-positive rate of that grade.
    """
    called = prediction >= cutoff
    y = np.asarray(y).astype(bool)
    grade = np.asarray(grade)
    out = {
        "auc": float(roc_auc_score(y, prediction)),
        "precision": float(y[called].mean()) if called.any() else np.nan,
        "recall": float(called[y].mean()),
    }
    for g in np.unique(grade):
        out[f"called_positive_grade_{g}"] = float(called[grade == g].mean())
    return out


def box_bootstrap_auc(
    prediction: np.ndarray,
    y: np.ndarray,
    boxes: np.ndarray,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float, np.ndarray]:
    """95 % percentile interval of the AUC by drawing boxes with replacement.

    The predictions are not refitted, only resampled (`docs/cv_explained.md`, section 6). A
    draw without both classes has no AUC and is skipped. Out: (lower, upper, all AUCs).
    """
    boxes = np.asarray(boxes)
    names = np.unique(boxes)
    members = {b: np.flatnonzero(boxes == b) for b in names}
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_resamples):
        drawn = rng.choice(names, size=len(names), replace=True)
        index = np.concatenate([members[b] for b in drawn])
        if len(np.unique(y[index])) == 2:
            aucs.append(roc_auc_score(y[index], prediction[index]))
    aucs = np.array(aucs)
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5)), aucs

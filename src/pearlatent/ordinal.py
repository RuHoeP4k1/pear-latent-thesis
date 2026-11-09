"""Ordinal (cumulative-link) model of the grade on one PLS-DA prediction per fruit.

Why: the four grades lie in order on the out-of-fold prediction of the binary PLS-DA model
(`binary_23`, notebook 08), but one PLS model per grade does poorly on the middle grades (notebook
11, `docs/methods_plsda.md` section 8). A cumulative logit model on that one number keeps the
order: P(grade <= j) = logistic(c_j - beta * score), with three cut-points c_0 < c_1 < c_2 and
one slope beta. The grade probabilities are the differences of these cumulative ones.

The model is fitted with `statsmodels` `OrderedModel` (`distr="logit"`, no constant: the
cut-points take its place) on the training boxes and applied to the held-out boxes, with the
same box folds as the PLS-DA.
"""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
from statsmodels.miscmodels.ordinal_model import OrderedModel

from pearlatent.splits import assert_no_group_leakage


def fit_ordinal(score: np.ndarray, grade: np.ndarray, classes: tuple):
    """Fit the cumulative logit model of `grade` on `score`.

    In: one score per fruit, the grade per fruit, the ordered classes (for example
    (0, 1, 2, 3)). Out: the fitted statsmodels result; `result.predict(exog=...)` gives the
    class probabilities of new fruit. Raises if a class is missing, because the cut-point of a
    missing class cannot be estimated.
    """
    missing = set(classes) - set(np.unique(grade))
    if missing:
        raise ValueError(
            f"classes without fruit in the training set: {sorted(missing)}"
        )
    model = OrderedModel(grade.astype(np.int64), score.reshape(-1, 1), distr="logit")
    with warnings.catch_warnings():
        # statsmodels warns about its optimiser; convergence is checked below instead.
        warnings.simplefilter("ignore")
        result = model.fit(method="bfgs", disp=False, maxiter=1000)
    if not result.mle_retvals["converged"]:
        raise RuntimeError("ordinal model did not converge")
    return result


def slope_and_cutpoints(result) -> tuple[float, np.ndarray]:
    """The slope beta and the three cut-points c_j (on the score scale times beta).

    statsmodels stores the cut-points as the first one plus log-increments; this converts
    them back. The class boundary on the score itself lies at c_j / beta.
    """
    params = np.asarray(result.params)
    cutpoints = result.model.transform_threshold_params(params)[1:-1]
    return float(params[0]), cutpoints


def cross_validated_ordinal(
    score: np.ndarray,
    grade: np.ndarray,
    boxes: np.ndarray,
    folds: pl.DataFrame,
    classes: tuple = (0, 1, 2, 3),
) -> np.ndarray:
    """Out-of-fold class probabilities, averaged over the repeats of the box folds.

    In: one score per fruit (here the saved out-of-fold PLS-DA prediction), the grade, the box
    of every fruit, the fold table (columns repeat, fold, box). For every repeat and fold the
    model is fitted on the training boxes and predicts the held-out boxes. Out: array
    (fruit, classes), the mean over the repeats.
    """
    boxes = np.asarray(boxes)
    repeats = sorted(folds["repeat"].unique().to_list())
    out = np.full((len(repeats), score.size, len(classes)), np.nan)
    for r_index, repeat in enumerate(repeats):
        partition = folds.filter(pl.col("repeat") == repeat)
        fold_of_box = dict(zip(partition["box"], partition["fold"], strict=True))
        fold_of_fruit = np.array([fold_of_box[b] for b in boxes])
        for fold in sorted(set(fold_of_fruit)):
            held_out = fold_of_fruit == fold
            sides = pl.DataFrame(
                {"box": boxes, "side": np.where(held_out, "test", "train")}
            )
            assert_no_group_leakage(sides, "box", "side")
            result = fit_ordinal(score[~held_out], grade[~held_out], classes)
            out[r_index, held_out] = result.predict(exog=score[held_out].reshape(-1, 1))
    if np.isnan(out).any():
        raise AssertionError("some fruit were not predicted in some repeat")
    return out.mean(axis=0)

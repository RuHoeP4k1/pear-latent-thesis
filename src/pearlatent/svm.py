"""Support vector machine discriminant analysis (SVM-DA) under design A, as a check on PLS-DA.

Why (Ruben, 20 November 2026; reopens "SVM not added" of 2 November): one number per fruit,
the SD of the interior CT cells, separates grade 0 from grade 1 (AUC 0.82), while every linear
PLS-DA row does not (0.59 to 0.72). A weighted sum of cell values cannot form a spread, so the
question is whether a non-linear boundary finds something the linear model misses. One row, on
the latent height-radius mean + SD features (320 numbers per fruit), the same 20 × 4 box folds.
If it clearly beats PLS-DA on the same features, there is non-linear information that our
features do not yet capture.

Model: scikit-learn `SVC` with a radial basis function (RBF) kernel,
k(a, b) = exp(−gamma · ‖a − b‖²). Two settings:

- C: how hard a misclassified training fruit is penalised. Small C gives a smooth boundary that
  tolerates errors, large C bends the boundary to fit every training fruit.
- gamma: how far the influence of one training fruit reaches. Small gamma gives an almost
  linear boundary, large gamma a very local one. Given here as a factor times 1 / (number of
  features); on standardised features that product is scikit-learn's default `gamma="scale"`.

Standardisation: every feature is scaled to mean 0 and SD 1 on the training fruit of each fold
(a `Pipeline`, so the held-out boxes never enter the means and SDs). PLS-DA on the same features
centres only; the SVM needs scaling, because the RBF kernel uses distances between fruit and a
feature with a large spread would otherwise dominate them.

The score of a fruit is the SVM decision value (signed distance to the boundary), not a
probability. The AUC only uses the order of the scores, so this is enough.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from pearlatent.splits import assert_no_group_leakage

# Fixed before any result was seen (20 November 2026). Four values of C over three orders of
# magnitude, three values of gamma around scikit-learn's default (factor 1).
C_VALUES = (0.1, 1.0, 10.0, 100.0)
GAMMA_FACTORS = (0.1, 1.0, 10.0)


def settings_grid(
    c_values=C_VALUES, gamma_factors=GAMMA_FACTORS
) -> list[tuple[float, float]]:
    """All (C, gamma factor) pairs, C varying slowest. Out: a list of 12 pairs by default."""
    return [(c, g) for c in c_values for g in gamma_factors]


def svm_model(c: float, gamma_factor: float, n_features: int):
    """Standardisation followed by an RBF support vector machine with these settings."""
    return make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=c, gamma=gamma_factor / n_features),
    )


def cross_validated_decisions(
    x: np.ndarray,
    y: np.ndarray,
    boxes: np.ndarray,
    folds: pl.DataFrame,
    settings: list[tuple[float, float]],
) -> np.ndarray:
    """Out-of-fold SVM decision values for every repeat, fruit and setting.

    In: features `x` (fruit × features), the 0/1 label `y`, the box of every fruit, the fold
    table (columns repeat, fold, box) and the list of (C, gamma factor) pairs. Out: an array
    (repeats, fruit, settings). For every repeat and fold one model per setting is fitted on
    the other 9 boxes and scores the 3 held-out boxes, so every fruit is scored once per repeat,
    by models that never saw its box. Raises on the same fold problems as
    `pearlatent.plsda.cross_validated_predictions`.
    """
    boxes = np.asarray(boxes)
    y = np.asarray(y)
    repeats = sorted(folds["repeat"].unique().to_list())
    out = np.full((len(repeats), len(y), len(settings)), np.nan)
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
            for s_index, (c, gamma_factor) in enumerate(settings):
                model = svm_model(c, gamma_factor, x.shape[1])
                model.fit(x[~held_out], y[~held_out])
                out[r_index, held_out, s_index] = model.decision_function(x[held_out])
    if np.isnan(out).any():
        raise AssertionError("some fruit were not scored in some repeat")
    return out

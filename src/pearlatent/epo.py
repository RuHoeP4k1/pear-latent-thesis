"""External parameter orthogonalisation (EPO) with the healthy fruit as the nuisance set.

EPO (Roger, Chauchard & Bellon-Maurel 2003): remove from every row of X the subspace spanned by
the first principal components of a matrix D that holds only nuisance variation,
X* = X (I − P Pᵀ). In the textbook form D holds the same objects measured while the external
parameter varies (`rchemo::eposvd`; course Chemometrics 2020, chapter 2). We have no such paired
measurements, so D is the healthy (grade 0) training fruit, centred on their own mean: they have
no defect, so their main variation is size, shape, stalk and orchard. A variant of EPO, to be
stated as such (question for Wouter Saeys).

Why (20 November 2026, scratch on the 315 open 2526 fruit): the first principal component of
the healthy latent grids carries 22 % of their variation, follows body size at r = 0.97 and stalk
length at r = −0.83, and points the same way as the latent's covariance with body size (cosine
0.99). So the PCA finds the size axis without the stalk and size measures, which Ruben judged
not accurate enough to use as covariates (2 November 2026). Removing it kept the AUC of every
latent row and removed the stalk and size link of the latent height-radius rows (check C).

For the counterfactuals: a PLS model fitted on X* has its regression vector b in the row space of
X*, so b is orthogonal to every removed direction. A step along b then leaves the fruit's amount of
the size pattern unchanged, and the model gives the same prediction on X and on X*.

The directions are fitted, so they come from the healthy fruit of the training boxes only.
"""

from __future__ import annotations

import numpy as np

N_DIRECTIONS = 1  # Ruben, 20 November 2026: PC 1 only; more PCs began to cost AUC


def fit_epo_directions(
    x_train: np.ndarray, healthy_train: np.ndarray, n_directions: int = N_DIRECTIONS
) -> np.ndarray:
    """The nuisance directions: the first principal components of the healthy training fruit.

    In: features (fruit, features) of the training fruit, a boolean per training fruit (True =
    grade 0), the number of directions. Out: an array (features, n_directions) with orthonormal
    columns. Raises when fewer than two healthy fruit are given.
    """
    x_train = np.asarray(x_train, dtype=float)
    healthy_train = np.asarray(healthy_train, dtype=bool)
    if healthy_train.sum() < 2:
        raise ValueError("EPO needs at least two healthy training fruit")
    deviations = x_train[healthy_train] - x_train[healthy_train].mean(axis=0)
    # Rows of vt are the principal component directions, largest variance first.
    _, _, vt = np.linalg.svd(deviations, full_matrices=False)
    return vt[:n_directions].T


def remove_directions(x: np.ndarray, directions: np.ndarray) -> np.ndarray:
    """X* = X − (X P) Pᵀ: every row without its part along the columns of P.

    In: features (fruit, features) or one row, directions P (features, k) with orthonormal
    columns. Out: a new array of the same shape as `x`.
    """
    x = np.asarray(x, dtype=float)
    return x - (x @ directions) @ directions.T


def fit_stretch_direction(
    x_train: np.ndarray,
    healthy_train: np.ndarray,
    stretch_train: np.ndarray,
    volume_train: np.ndarray,
) -> np.ndarray:
    """EPO direction from the measured stretch factor, at equal real size (27 November 2026).

    In: features (fruit, features) of a fold's training fruit, a boolean per fruit (True =
    grade 0), the stretch factor s = 128 / original length and the real body volume (cm³) of
    each fruit (`pearlatent.real_size`). Out: an array (features, 1) with a unit column.

    On the healthy training fruit, every feature is regressed on the standardised stretch
    factor and the standardised real volume together (with an intercept). The coefficients of
    the stretch factor form one vector: the change of each feature per standard deviation of
    stretch when real size is held equal. That vector, scaled to length 1, is the direction
    removed. Why both: s and real volume correlate (−0.52 on the healthy 2526 fruit,
    `docs/methods_plsda.md` section 20); regressing on s alone would also remove part of the
    real-size pattern, which is biology (section 21) and stays in. Why healthy fruit only:
    real size goes with the defect, so on all fruit the stretch coefficients could absorb
    defect signal through the correlations.
    """
    x_train = np.asarray(x_train, dtype=float)
    healthy_train = np.asarray(healthy_train, dtype=bool)
    if healthy_train.sum() < 4:
        raise ValueError(
            "the stretch direction needs at least four healthy training fruit"
        )
    x_h = x_train[healthy_train]
    s = np.asarray(stretch_train, dtype=float)[healthy_train]
    v = np.asarray(volume_train, dtype=float)[healthy_train]
    design = np.column_stack(
        [np.ones_like(s), (s - s.mean()) / s.std(), (v - v.mean()) / v.std()]
    )
    # One least-squares fit for all features at once: coefficients has shape (3, features).
    coefficients, *_ = np.linalg.lstsq(design, x_h, rcond=None)
    direction = coefficients[1]
    return (direction / np.linalg.norm(direction))[:, None]

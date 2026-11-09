"""Counterfactual images (charter, Extensions; notebook 17).

Plan (Ruben, 7 and 9 November 2026): move a fruit's latent `mu` along the direction of a PLS-DA
model on the full 32³ latent grid and decode the result, for two directions on the same fruit:
the cavity-free browning model and the general `binary_23` model. First step, here: which fruit.

`draw_example_fruit`: 2 grade 3 and 2 grade 0 fruit, all cavity-free, from 4 different boxes,
with a fixed seed. Cavity-free so that both directions apply to the same fruit and a change in
the image cannot come from an existing cavity. Ruben looks at their CT slices before any decode.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from sklearn.cross_decomposition import PLSRegression

DRAW_SEED = 369
N_PER_GROUP = 2


def draw_example_fruit(
    fruit: pl.DataFrame,
    n_per_group: int = N_PER_GROUP,
    seed: int = DRAW_SEED,
    exclude: tuple[str, ...] = (),
) -> pl.DataFrame:
    """The example fruit: `n_per_group` of grade 3 and of grade 0, every fruit from its own box.

    In: a table with `fruit_id`, `box`, `browning`, `cavity`, `defective` (the open fruit after the
    faulty volumes). Out: the drawn rows with a `group` column ("grade 3" first, then "grade 0").

    Grade 3 = browning 3 and cavity 0; grade 0 = defective 0. Each candidate list is sorted by
    fruit_id (so the row order of `fruit` does not matter) and shuffled with one generator
    `default_rng(seed)`; walking down the shuffled list, a fruit is taken when its box is not used
    yet. Grade 3 first, because it has fewer candidates and boxes. Fruit in `exclude` are
    skipped (rejected by Ruben on their CT), so the next fruit of the same shuffled list takes
    their place.
    """
    rng = np.random.default_rng(seed)
    groups = {
        "grade 3": (pl.col("browning") == 3) & (pl.col("cavity") == 0),
        "grade 0": pl.col("defective") == 0,
    }
    used_boxes: set[str] = set()
    picked = []
    for group, condition in groups.items():
        candidates = fruit.filter(condition).sort("fruit_id")
        print(
            f"{group}: {candidates.height} candidates in {candidates['box'].n_unique()} boxes"
        )
        taken = 0
        for index in rng.permutation(candidates.height):
            row = candidates.row(int(index), named=True)
            if row["box"] in used_boxes or row["fruit_id"] in exclude:
                continue
            picked.append({**row, "group": group})
            used_boxes.add(row["box"])
            taken += 1
            if taken == n_per_group:
                break
        if taken < n_per_group:
            raise ValueError(f"{group}: not enough boxes left for {n_per_group} fruit")
    return pl.DataFrame(picked).select("group", *fruit.columns)


@dataclass(frozen=True)
class Direction:
    """A fitted PLS-DA model reduced to what a counterfactual needs.

    For a centred-only model (`scale=False`) the prediction of a fruit is
    y_mean + (x - x_mean) · b, with b the regression vector (one number per cell).
    """

    coefficients: np.ndarray  # b, one per feature
    x_mean: np.ndarray
    y_mean: float
    n_components: int

    def predict(self, x: np.ndarray) -> np.ndarray | float:
        """Prediction of one row or of rows (fruit, features)."""
        return self.y_mean + (np.asarray(x) - self.x_mean) @ self.coefficients


def plsda_direction(x: np.ndarray, y: np.ndarray, n_components: int) -> Direction:
    """Fit one PLS-DA model (centred only) on all given fruit and keep its regression vector.

    In: features (fruit, cells), 0/1 labels, number of LVs. Out: a `Direction`. One fit on all
    12 open boxes (HANDOFF, 7 November 2026: the general model, not averaged fold models).
    """
    model = PLSRegression(n_components=n_components, scale=False).fit(
        x, np.asarray(y, dtype=float)
    )
    # scikit-learn stores coef_ as (targets, features); one target here.
    coefficients = np.asarray(model.coef_).reshape(-1)
    direction = Direction(
        coefficients, model._x_mean, float(model._y_mean[0]), n_components
    )
    # Check: the reduced form gives the same prediction as scikit-learn.
    if not np.allclose(direction.predict(x), model.predict(x).ravel(), atol=1e-6):
        raise AssertionError("regression vector does not reproduce the predictions")
    return direction


def shift_to_score(row: np.ndarray, direction: Direction, target: float) -> np.ndarray:
    """The fruit's features moved along b until the model predicts `target`.

    In: one row (cells), the direction, the target score. Out: the moved row.
    Change = (target - current prediction) × b / (b · b). This is the smallest change (in the
    Euclidean sense) that gives exactly that prediction; every other change with the same
    prediction adds a part that the model does not see. Example: b = (2, 0), prediction 0.2,
    target 1.0: change = 0.8 × (2, 0) / 4 = (0.4, 0), and the prediction rises by 2 × 0.4 = 0.8.
    """
    b = direction.coefficients
    step = (target - direction.predict(row)) / (b @ b)
    return np.asarray(row, dtype=float) + step * b

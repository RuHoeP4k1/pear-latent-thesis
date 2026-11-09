"""Nuisance checks on out-of-fold scores (`docs/nuisance_check.md`, section 3).

Every function takes only the out-of-fold score of each fruit plus its box, label and
descriptors, so the same checks apply to every model row whatever is inside the model.

- `within_box_auc` (check A): the share of pairs (one non-consumable, one consumable fruit of
  the same box) in which the non-consumable fruit gets the higher score.
- `box_bootstrap` : 95 % interval of any of these statistics by drawing boxes with
  replacement (charter rule 4).
- `within_between_logit` (not used; dropped by Ruben on 7 November 2026, kept for the record):
  `binary_23` on the box mean of the score and the fruit's
  deviation from it. A logistic regression with standard errors clustered by box, not the
  random-intercept model the note asks for: statsmodels has no frequentist mixed logistic
  model and R's lme4 is not installed (2 November 2026). With the box mean as a predictor the
  two models estimate the same within-box effect (the Mundlak argument); to be repeated with
  lme4 `glmer` if Ruben wants the mixed model itself.
- `score_icc` (reported beside check A): share of the variation of the score between boxes once the defective
  grade is accounted for (linear mixed model, random intercept per box).
- `healthy_fruit_correlations` (check C): on grade 0 fruit only, Spearman correlation of the
  score (minus its box median) with stalk length and body size, and the difference in median
  score between fruit with and without a stalk.

Check D (descriptors as covariates) is not built: Ruben, 2 November 2026, does not consider
the stalk and size measures in the rescaled grid accurate enough to serve as covariates.
"""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import spearmanr

BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 369


def within_box_auc(score: np.ndarray, y: np.ndarray, boxes: np.ndarray) -> float:
    """Check A. In: score, 0/1 label and box per fruit. Out: the within-box AUC.

    Pairs are formed only inside a box; a tie counts one half, as in the ordinary AUC. Boxes
    with only one class add no pairs.
    """
    score, y, boxes = np.asarray(score), np.asarray(y), np.asarray(boxes)
    wins = 0.0
    pairs = 0
    for box in np.unique(boxes):
        in_box = boxes == box
        positive = score[in_box & (y == 1)]
        negative = score[in_box & (y == 0)]
        if positive.size == 0 or negative.size == 0:
            continue
        difference = positive[:, None] - negative[None, :]
        wins += (difference > 0).sum() + 0.5 * (difference == 0).sum()
        pairs += difference.size
    return wins / pairs


def box_bootstrap(
    statistic,
    table: pl.DataFrame,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """95 % percentile interval of `statistic(table)` over draws of whole boxes.

    In: a function of a table and the table (needs a `box` column). A box drawn twice enters
    twice, under a new name, so that within-box statistics treat the copies as two boxes.
    Draws where the statistic cannot be computed (NaN) are skipped.
    """
    rng = np.random.default_rng(seed)
    names = table["box"].unique().sort().to_list()
    parts = {name: table.filter(pl.col("box") == name) for name in names}
    values = []
    for _ in range(n_resamples):
        drawn = rng.choice(names, size=len(names), replace=True)
        sample = pl.concat(
            [
                parts[name].with_columns(box=pl.lit(f"{name}#{k}"))
                for k, name in enumerate(drawn)
            ]
        )
        value = statistic(sample)
        if np.isfinite(value):
            values.append(value)
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def split_score(table: pl.DataFrame) -> pl.DataFrame:
    """Add `score_box_mean` and `score_within` (the score minus its box mean)."""
    return table.with_columns(
        score_box_mean=pl.col("score").mean().over("box")
    ).with_columns(score_within=pl.col("score") - pl.col("score_box_mean"))


def within_between_logit(table: pl.DataFrame) -> pl.DataFrame:
    """Not used (see the module docstring). In: a table with `score`, `box`, `binary_23`. Out: one row per predictor.

    Both parts of the score are divided by the standard deviation of the score over all fruit,
    so the two coefficients are on the same scale (change in log-odds per standard deviation
    of the score). Standard errors are clustered by box.
    """
    data = split_score(table)
    sd = data["score"].std()
    data = data.with_columns(
        box_mean_sd=pl.col("score_box_mean") / sd, within_sd=pl.col("score_within") / sd
    ).to_pandas()
    groups = data["box"].astype("category").cat.codes
    fit = smf.glm(
        "binary_23 ~ box_mean_sd + within_sd", data=data, family=sm.families.Binomial()
    ).fit(cov_type="cluster", cov_kwds={"groups": groups})
    interval = fit.conf_int()
    return pl.DataFrame(
        {
            "predictor": ["box mean of the score", "within-box deviation"],
            "log_odds_per_sd": [fit.params["box_mean_sd"], fit.params["within_sd"]],
            "lower_95": [interval.loc["box_mean_sd", 0], interval.loc["within_sd", 0]],
            "upper_95": [interval.loc["box_mean_sd", 1], interval.loc["within_sd", 1]],
            "p_value": [fit.pvalues["box_mean_sd"], fit.pvalues["within_sd"]],
        }
    )


def score_icc(table: pl.DataFrame) -> float:
    """Share of the score's variation between boxes, given the defective grade.

    Linear mixed model score ~ grade (as categories) + random intercept per box, fitted by
    restricted maximum likelihood. Out: box variance / (box variance + residual variance).
    """
    data = table.select("score", "box", "defective").to_pandas()
    with warnings.catch_warnings():
        # A box variance near 0 gives a boundary warning; the value 0 is then the answer.
        warnings.simplefilter("ignore")
        fit = smf.mixedlm("score ~ C(defective)", data, groups=data["box"]).fit(
            reml=True
        )
    box_variance = float(fit.cov_re.iloc[0, 0])
    return box_variance / (box_variance + float(fit.scale))


def healthy_fruit_correlations(table: pl.DataFrame) -> dict:
    """Check C. In: all fruit with `score`, `box`, `defective`, `stalk_length`,
    `body_voxels`, `stalk_present`. Out: the statistics on the grade 0 fruit.

    The score is taken minus its box median (over the healthy fruit of that box), so that
    differences between boxes do not create a correlation on their own.
    """
    healthy = table.filter(pl.col("defective") == 0).with_columns(
        score_in_box=pl.col("score") - pl.col("score").median().over("box")
    )
    with_stalk = healthy.filter(pl.col("stalk_present"))["score_in_box"]
    without_stalk = healthy.filter(~pl.col("stalk_present"))["score_in_box"]
    return {
        "n_healthy": healthy.height,
        "spearman_stalk_length": float(
            spearmanr(healthy["score_in_box"], healthy["stalk_length"]).statistic
        ),
        "spearman_body_voxels": float(
            spearmanr(healthy["score_in_box"], healthy["body_voxels"]).statistic
        ),
        "median_difference_stalk_minus_no_stalk": float(
            with_stalk.median() - without_stalk.median()
        )
        if with_stalk.len() and without_stalk.len()
        else np.nan,
    }

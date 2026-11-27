import numpy as np
import polars as pl

from pearlatent.nuisance import (
    box_bootstrap,
    healthy_fruit_correlations,
    healthy_stretch_correlations,
    partial_spearman,
    score_icc,
    within_between_logit,
    within_box_auc,
)


def _table(box_effect: float, fruit_effect: float, seed: int = 0) -> pl.DataFrame:
    """12 boxes of 30 fruit. Boxes differ in their share of defects; the score follows the
    box share by `box_effect` and the fruit's own label by `fruit_effect`."""
    rng = np.random.default_rng(seed)
    boxes = np.repeat([f"B{k:02d}" for k in range(12)], 30)
    share = np.repeat(np.linspace(0.1, 0.7, 12), 30)
    y = (rng.random(boxes.size) < share).astype(int)
    score = box_effect * share + fruit_effect * y + rng.normal(0, 0.3, boxes.size)
    return pl.DataFrame(
        {
            "box": boxes,
            "binary_23": y,
            "defective": y * 2,
            "score": score,
            "stalk_length": rng.integers(0, 30, boxes.size),
            "body_voxels": rng.normal(150_000, 10_000, boxes.size),
            "stalk_present": rng.random(boxes.size) < 0.7,
        }
    )


def test_within_box_auc_is_chance_when_the_score_only_ranks_boxes():
    t = _table(box_effect=5.0, fruit_effect=0.0)
    assert abs(within_box_auc(t["score"], t["binary_23"], t["box"]) - 0.5) < 0.06


def test_within_box_auc_is_high_when_the_score_follows_the_fruit():
    t = _table(box_effect=0.0, fruit_effect=2.0)
    assert within_box_auc(t["score"], t["binary_23"], t["box"]) > 0.95


def test_within_box_auc_counts_only_pairs_inside_a_box():
    # Box A: positive 1.0 beats negative 0.0. Box B: positive 0.2 loses to negative 0.5.
    # Across boxes 1.0 > 0.5 would count, but it must not.
    score = np.array([1.0, 0.0, 0.2, 0.5])
    y = np.array([1, 0, 1, 0])
    boxes = np.array(["A", "A", "B", "B"])
    assert within_box_auc(score, y, boxes) == 0.5


def test_logit_separates_box_and_fruit_parts():
    # fruit_effect small enough that the classes overlap (no perfect separation).
    rows = within_between_logit(_table(box_effect=0.0, fruit_effect=0.3))
    within = rows.filter(pl.col("predictor") == "within-box deviation")
    assert within["lower_95"][0] > 0


def test_icc_is_near_zero_without_box_effect_and_large_with_one():
    assert score_icc(_table(box_effect=0.0, fruit_effect=1.0)) < 0.05
    assert score_icc(_table(box_effect=5.0, fruit_effect=1.0)) > 0.5


def test_box_bootstrap_interval_contains_the_estimate():
    t = _table(box_effect=0.0, fruit_effect=1.0)

    def stat(x):
        return within_box_auc(x["score"], x["binary_23"], x["box"])

    low, high = box_bootstrap(stat, t, n_resamples=200)
    assert low <= stat(t) <= high


def test_healthy_correlations_use_grade_0_only():
    out = healthy_fruit_correlations(_table(box_effect=0.0, fruit_effect=1.0))
    assert out["n_healthy"] < 360
    assert abs(out["spearman_stalk_length"]) < 0.25


def _stretch_table(score_follows: str, seed: int = 1) -> pl.DataFrame:
    """300 healthy fruit in 10 boxes. Real volume and stretch factor correlate (about −0.5),
    as on the 2526 fruit; the score follows one of them plus noise."""
    rng = np.random.default_rng(seed)
    n = 300
    volume = rng.normal(0, 1, n)
    stretch = -0.5 * volume + rng.normal(0, 0.87, n)
    driver = stretch if score_follows == "stretch" else volume
    return pl.DataFrame(
        {
            "box": np.repeat([f"B{k}" for k in range(10)], 30),
            "defective": np.zeros(n, dtype=int),
            "score": driver + rng.normal(0, 0.5, n),
            "stretch_factor": 0.6 + 0.08 * stretch,
            "body_volume_cm3": 160 + 30 * volume,
        }
    )


def test_partial_spearman_removes_the_shared_part():
    """y follows x only; control is correlated with x. Plain correlation of y with control
    is clearly negative, the partial is near 0."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=500)
    control = -0.6 * x + rng.normal(0, 0.8, 500)
    y = x + rng.normal(0, 0.3, 500)
    assert np.corrcoef(y, control)[0, 1] < -0.4
    assert abs(partial_spearman(y, control, x)) < 0.1
    assert partial_spearman(y, x, control) > 0.8


def test_healthy_stretch_correlations_tell_stretch_from_volume():
    follows_stretch = healthy_stretch_correlations(_stretch_table("stretch"))
    assert follows_stretch["partial_stretch_given_volume"] > 0.6
    assert abs(follows_stretch["partial_volume_given_stretch"]) < 0.15
    assert follows_stretch["spearman_real_volume"] < -0.2  # the trap: plain correlation
    follows_volume = healthy_stretch_correlations(_stretch_table("volume"))
    assert follows_volume["partial_volume_given_stretch"] > 0.6
    assert abs(follows_volume["partial_stretch_given_volume"]) < 0.15

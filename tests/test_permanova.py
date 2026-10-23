import numpy as np
import pytest

from pearlatent.permanova import (
    permanova,
    permdisp,
    permuted_labels,
    squared_distances,
)


def test_squared_distances():
    x = np.array([[0.0, 0.0], [3.0, 4.0], [0.0, 1.0]])
    d2 = squared_distances(x)
    assert d2[0, 1] == pytest.approx(25)
    assert d2[0, 2] == pytest.approx(1)
    assert np.allclose(d2, d2.T) and np.all(np.diag(d2) == 0)


def test_pseudo_f_and_r2_match_anova_in_one_dimension():
    # For one variable and Euclidean distance, PERMANOVA's pseudo-F equals the ANOVA F.
    x = np.array([1.0, 2.0, 3.0, 7.0, 8.0, 9.0])[:, None]
    labels = ["a", "a", "a", "b", "b", "b"]
    result = permanova(squared_distances(x), labels, n_permutations=99)
    # Between: 2 groups, means 2 and 8, grand mean 5: SS_between = 3·9 + 3·9 = 54.
    # Within: 2 + 2 = 4. F = (54 / 1) / (4 / 4) = 54; R² = 54 / 58.
    assert result["pseudo_F"] == pytest.approx(54)
    assert result["R2"] == pytest.approx(54 / 58)


def test_separated_groups_small_p_mixed_groups_large_p():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(40, 5))
    labels = np.repeat(["a", "b"], 20)
    shifted = x + (labels == "b")[:, None] * 3.0
    assert permanova(squared_distances(shifted), labels, n_permutations=199)["p"] < 0.01
    assert permanova(squared_distances(x), labels, n_permutations=199)["p"] > 0.05


def test_permutations_stay_within_strata():
    codes = np.array([0, 1, 0, 1, 2, 2, 3, 3])
    strata = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])
    for row in permuted_labels(codes, strata, 50, seed=0):
        assert sorted(row[:4]) == [0, 0, 1, 1]
        assert sorted(row[4:]) == [2, 2, 3, 3]


def test_restricted_permutation_removes_a_stratum_effect():
    # The grade has no effect of its own; box B is shifted and holds most high grades.
    rng = np.random.default_rng(2)
    box = np.repeat(["A", "B"], 30)
    grade = np.concatenate(
        [rng.choice([0, 1], 30, p=[0.8, 0.2]), rng.choice([0, 1], 30, p=[0.2, 0.8])]
    )
    x = rng.normal(size=(60, 4)) + (box == "B")[:, None] * 4.0
    d2 = squared_distances(x)
    free = permanova(d2, grade, n_permutations=199)
    within_box = permanova(d2, grade, strata=box, n_permutations=199)
    assert free["p"] < 0.05  # the box difference looks like a grade effect
    assert within_box["p"] > 0.05  # not when grades are swapped only within a box


def test_permdisp_detects_spread():
    rng = np.random.default_rng(3)
    x = np.vstack([rng.normal(0, 1, (30, 3)), rng.normal(0, 4, (30, 3))])
    labels = np.repeat(["narrow", "wide"], 30)
    result = permdisp(x, labels, n_permutations=199)
    assert result["p"] < 0.01
    assert result["mean_distance"]["wide"] > result["mean_distance"]["narrow"]

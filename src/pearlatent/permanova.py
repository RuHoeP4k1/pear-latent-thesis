"""PERMANOVA and PERMDISP with permutations restricted within strata (`docs/exploration_plan.md`).

Written here rather than taken from scikit-bio because the grade tests need permutations
restricted to fruit of the same box, which scikit-bio's `permanova` does not offer.

PERMANOVA (Anderson 2001, Austral Ecology 26, 32-46), one factor with a groups:
    SS_total  = (1 / N) Σ_{i<j} d²_ij
    SS_within = Σ_g (1 / n_g) Σ_{i<j in g} d²_ij
    pseudo-F  = [(SS_total − SS_within) / (a − 1)] / [SS_within / (N − a)]
    R²        = (SS_total − SS_within) / SS_total
PERMDISP (Anderson 2006, Biometrics 62, 245-253): the Euclidean distance of every observation
to the centre of its group, compared between groups with the one-way analysis of variance F;
the same permutations of the group labels give its p-value.
"""

from __future__ import annotations

import numpy as np


def squared_distances(x: np.ndarray) -> np.ndarray:
    """Squared Euclidean distances between the rows of `x` (N × p), as an N × N matrix."""
    x = np.asarray(x, dtype=np.float64)
    norms = (x**2).sum(axis=1)
    d2 = norms[:, None] + norms[None, :] - 2 * x @ x.T
    # Rounding can leave tiny negative numbers; a distance is never negative.
    np.maximum(d2, 0, out=d2)
    np.fill_diagonal(d2, 0)
    return d2


def _codes(labels) -> tuple[np.ndarray, int]:
    """Labels as integer codes 0..a-1, and a."""
    _, codes = np.unique(np.asarray(labels), return_inverse=True)
    return codes, int(codes.max()) + 1


def _within_sum_of_squares(d2: np.ndarray, codes: np.ndarray, n_groups: int) -> float:
    """SS_within for one labelling (formula in the module docstring)."""
    onehot = np.zeros((codes.size, n_groups))
    onehot[np.arange(codes.size), codes] = 1
    # (onehotᵀ D² onehot)_gg = Σ over ordered pairs (i, j) in g of d²_ij = 2 Σ_{i<j}.
    pair_sums = ((onehot.T @ d2) * onehot.T).sum(axis=1) / 2
    return float((pair_sums / onehot.sum(axis=0)).sum())


def permuted_labels(
    codes: np.ndarray, strata, n_permutations: int, seed: int
) -> np.ndarray:
    """`n_permutations` shuffles of `codes` (rows), each label shuffled only within its stratum.

    `strata` None shuffles freely. The generator is seeded, so the shuffles are reproducible.
    """
    rng = np.random.default_rng(seed)
    out = np.empty((n_permutations, codes.size), dtype=codes.dtype)
    if strata is None:
        for k in range(n_permutations):
            out[k] = rng.permutation(codes)
        return out
    strata = np.asarray(strata)
    groups = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    for k in range(n_permutations):
        shuffled = codes.copy()
        for index in groups:
            shuffled[index] = codes[rng.permutation(index)]
        out[k] = shuffled
    return out


def permanova(
    d2: np.ndarray,
    labels,
    strata=None,
    n_permutations: int = 9999,
    seed: int = 369,
) -> dict:
    """One-factor PERMANOVA on squared distances `d2`.

    In: the N × N squared distances, one label per observation, optional strata (permutation
    only within a stratum). Out: pseudo-F, R², p and the counts.
    """
    codes, a = _codes(labels)
    n = codes.size
    ss_total = float(d2[np.triu_indices(n, 1)].sum() / n)

    def pseudo_f(c: np.ndarray) -> float:
        ss_within = _within_sum_of_squares(d2, c, a)
        return ((ss_total - ss_within) / (a - 1)) / (ss_within / (n - a))

    observed = pseudo_f(codes)
    permutations = permuted_labels(codes, strata, n_permutations, seed)
    null = np.array([pseudo_f(c) for c in permutations])
    ss_within = _within_sum_of_squares(d2, codes, a)
    return {
        "pseudo_F": observed,
        "R2": (ss_total - ss_within) / ss_total,
        "p": float((np.sum(null >= observed) + 1) / (n_permutations + 1)),
        "n": n,
        "groups": a,
        "permutations": n_permutations,
    }


def permdisp(
    x: np.ndarray,
    labels,
    strata=None,
    n_permutations: int = 9999,
    seed: int = 369,
) -> dict:
    """PERMDISP: do the groups differ in spread around their own centre?

    In: the observations `x` (N × p), labels, optional strata. Out: F of the one-way analysis
    of variance of the distances to the group centres, p, and the mean distance per group.
    """
    x = np.asarray(x, dtype=np.float64)
    codes, a = _codes(labels)
    names = np.unique(np.asarray(labels))
    centres = np.stack([x[codes == g].mean(axis=0) for g in range(a)])
    z = np.linalg.norm(x - centres[codes], axis=1)

    def anova_f(c: np.ndarray) -> float:
        means = np.bincount(c, weights=z, minlength=a) / np.bincount(c, minlength=a)
        counts = np.bincount(c, minlength=a)
        between = (counts * (means - z.mean()) ** 2).sum() / (a - 1)
        within = ((z - means[c]) ** 2).sum() / (z.size - a)
        return between / within

    observed = anova_f(codes)
    # The distances to the centres stay fixed and only the labels move (permutation of the
    # labels of the distances, as in the simplest version of Anderson's test).
    permutations = permuted_labels(codes, strata, n_permutations, seed)
    null = np.array([anova_f(c) for c in permutations])
    return {
        "F": float(observed),
        "p": float((np.sum(null >= observed) + 1) / (n_permutations + 1)),
        "mean_distance": {str(names[g]): float(z[codes == g].mean()) for g in range(a)},
    }

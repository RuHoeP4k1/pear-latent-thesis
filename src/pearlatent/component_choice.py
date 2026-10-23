"""Choice of the number of PLS latent variables (`docs/cv_explained.md`, section 4, item 3).

Rule decided by Ruben on 23 October 2026, before any cross-validation curve was computed: the
randomisation test of van der Voet (1994, Chemometrics and Intelligent Laboratory Systems 25,
313-323) with the box, not the fruit, as the unit whose sign is flipped. All 2^B sign
combinations are enumerated, so the test is exact and needs no seed.
"""

from __future__ import annotations

import itertools

import numpy as np

# Default cut-off of the SAS PLS procedure for this test.
P_CUTOFF = 0.10


def box_sums(values: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Sum `values` (one per fruit) within each box. Out: one number per box, boxes sorted."""
    values = np.asarray(values, dtype=float)
    boxes = np.asarray(boxes)
    return np.array([values[boxes == b].sum() for b in np.unique(boxes)])


def sign_flip_p_value(box_differences: np.ndarray) -> tuple[float, float]:
    """Exact one-sided sign-flip test on per-box differences.

    In: D_b, one number per box (positive when the smaller model predicts worse).
    Out: (T, p) with T = Σ D_b and p the share of the 2^B sign combinations whose sum is at
    least T. The observed combination is one of them, so p ≥ 1 / 2^B.
    """
    d = np.asarray(box_differences, dtype=float)
    observed = d.sum()
    # Every row of `signs` is one combination of +1 / -1 over the boxes.
    signs = np.array(list(itertools.product((1.0, -1.0), repeat=d.size)))
    flipped = signs @ d
    # A small tolerance so that the observed sum itself always counts despite rounding.
    p = float(np.mean(flipped >= observed - 1e-12 * max(1.0, abs(observed))))
    return float(observed), p


def choose_n_components(
    squared_errors: np.ndarray, boxes: np.ndarray, cutoff: float = P_CUTOFF
) -> dict:
    """Apply the rule of `docs/cv_explained.md`, section 4, item 3.

    In: `squared_errors`, shape (fruit, H): squared cross-validation error of each fruit with
    1, ..., H latent variables, already averaged over the repeats; `boxes`, the box of each
    fruit. Out: a dict with h_star (the minimum of the PRESS curve), the chosen h, the PRESS
    curve and, for every h < h_star, T(h) and p(h). Numbers of latent variables count from 1.
    """
    squared_errors = np.asarray(squared_errors, dtype=float)
    press = squared_errors.sum(axis=0)
    h_star = int(np.argmin(press)) + 1
    tests = {}
    chosen = h_star
    for h in range(1, h_star):
        differences = squared_errors[:, h - 1] - squared_errors[:, h_star - 1]
        t, p = sign_flip_p_value(box_sums(differences, boxes))
        tests[h] = {"T": t, "p": p}
    # The smallest h whose errors are not significantly larger than at h_star.
    for h in range(1, h_star):
        if tests[h]["p"] > cutoff:
            chosen = h
            break
    return {"h_star": h_star, "chosen": chosen, "press": press, "tests": tests}

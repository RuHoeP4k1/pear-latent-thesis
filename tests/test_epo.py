import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression

from pearlatent.epo import (
    fit_epo_directions,
    fit_stretch_direction,
    remove_directions,
)


def _healthy_with_size_axis(seed: int = 0):
    """60 fruit, 40 features; a strong 'size' pattern varies among all fruit, the defect shifts
    features 0 to 4 in the last 20 fruit only."""
    rng = np.random.default_rng(seed)
    size_pattern = rng.normal(size=40)
    size_pattern /= np.linalg.norm(size_pattern)
    size = rng.normal(scale=10.0, size=60)
    x = np.outer(size, size_pattern) + rng.normal(scale=0.5, size=(60, 40))
    y = np.zeros(60)
    y[40:] = 1
    x[40:, :5] += 2.0
    healthy = y == 0
    return x, y, healthy, size_pattern


def test_first_direction_is_the_main_healthy_axis():
    x, _, healthy, size_pattern = _healthy_with_size_axis()
    directions = fit_epo_directions(x, healthy)
    assert directions.shape == (40, 1)
    assert abs(directions[:, 0] @ size_pattern) > 0.95
    assert np.linalg.norm(directions[:, 0]) == pytest.approx(1.0)


def test_corrected_rows_have_no_part_along_the_directions():
    x, _, healthy, _ = _healthy_with_size_axis()
    directions = fit_epo_directions(x, healthy, n_directions=2)
    corrected = remove_directions(x, directions)
    np.testing.assert_allclose(corrected @ directions, 0.0, atol=1e-10)
    # One row works too.
    np.testing.assert_allclose(remove_directions(x[0], directions), corrected[0])


def test_directions_come_from_healthy_fruit_only():
    x, _, healthy, _ = _healthy_with_size_axis()
    changed = x.copy()
    changed[~healthy] += 1000.0  # defective fruit changed: directions must not move
    np.testing.assert_allclose(
        np.abs(fit_epo_directions(x, healthy)),
        np.abs(fit_epo_directions(changed, healthy)),
    )


def test_pls_regression_vector_is_orthogonal_to_the_removed_direction():
    x, y, healthy, _ = _healthy_with_size_axis()
    directions = fit_epo_directions(x, healthy)
    corrected = remove_directions(x, directions)
    model = PLSRegression(n_components=3, scale=False).fit(corrected, y)
    b = model.coef_.ravel()
    assert abs(b @ directions[:, 0]) < 1e-8 * np.linalg.norm(b)
    # So the model predicts the same on the raw and on the corrected rows.
    np.testing.assert_allclose(model.predict(x), model.predict(corrected), atol=1e-8)


def test_too_few_healthy_fruit_is_refused():
    x, _, _, _ = _healthy_with_size_axis()
    with pytest.raises(ValueError):
        fit_epo_directions(x, np.eye(60, dtype=bool)[0])


def _stretch_and_size(seed: int = 0):
    """200 healthy fruit, 30 features. A stretch pattern follows s, a different size pattern
    follows real volume; s and volume correlate about −0.5, as on the 2526 fruit."""
    rng = np.random.default_rng(seed)
    volume = rng.normal(size=200)
    stretch = -0.5 * volume + rng.normal(scale=0.87, size=200)
    stretch_pattern = np.zeros(30)
    stretch_pattern[:10] = 1.0
    size_pattern = np.zeros(30)
    size_pattern[5:20] = 1.0  # overlaps the stretch pattern on features 5 to 9
    x = (
        np.outer(stretch, stretch_pattern)
        + np.outer(volume, size_pattern)
        + rng.normal(scale=0.2, size=(200, 30))
    )
    return x, stretch, volume, stretch_pattern, size_pattern


def test_stretch_direction_finds_the_stretch_pattern_not_size():
    x, stretch, volume, stretch_pattern, size_pattern = _stretch_and_size()
    healthy = np.ones(200, dtype=bool)
    direction = fit_stretch_direction(
        x, healthy, 0.6 + 0.08 * stretch, 160 + 30 * volume
    )
    assert direction.shape == (30, 1)
    assert np.linalg.norm(direction) == pytest.approx(1.0)
    cosine = abs(direction[:, 0] @ stretch_pattern) / np.linalg.norm(stretch_pattern)
    assert cosine > 0.99
    cleaned = remove_directions(x, direction)
    # Regress the features on s and volume together, before and after the removal: the
    # stretch coefficients vanish; the volume coefficients lose only their part along the
    # removed direction (the overlap on features 5 to 9), here about 9 % of their length.
    design = np.column_stack([np.ones(200), stretch, volume])
    before, *_ = np.linalg.lstsq(design, x, rcond=None)
    after, *_ = np.linalg.lstsq(design, cleaned, rcond=None)
    assert np.linalg.norm(after[1]) < 0.05 * np.linalg.norm(before[1])
    assert np.linalg.norm(after[2]) > 0.85 * np.linalg.norm(before[2])


def test_stretch_direction_uses_healthy_fruit_only():
    x, stretch, volume, *_ = _stretch_and_size()
    healthy = np.zeros(200, dtype=bool)
    healthy[:100] = True
    x_changed = x.copy()
    x_changed[100:] += 50.0  # the unhealthy fruit must not matter
    a = fit_stretch_direction(x, healthy, stretch, volume)
    b = fit_stretch_direction(x_changed, healthy, stretch, volume)
    assert np.allclose(a, b)
    with pytest.raises(ValueError):
        fit_stretch_direction(x, np.zeros(200, dtype=bool), stretch, volume)

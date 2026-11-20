import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression

from pearlatent.epo import fit_epo_directions, remove_directions


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

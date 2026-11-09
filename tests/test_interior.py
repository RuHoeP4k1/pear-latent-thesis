import numpy as np
import pytest

from pearlatent.interior import interior_cells, interior_sd


def _ball(radius: float = 8.0) -> np.ndarray:
    x, y, z = np.meshgrid(*(np.arange(32) for _ in range(3)), indexing="ij")
    return np.hypot(np.hypot(x - 15.5, y - 15.5), z - 15.5) < radius


def test_one_layer_removes_the_cells_on_the_surface():
    body = _ball()
    inside = interior_cells(body, 1)
    assert inside.sum() < body.sum()
    assert not (inside & ~body).any()
    # Every removed cell touches the outside across a face.
    removed = body & ~inside
    outside = ~body
    touches = np.zeros_like(body)
    for axis in range(3):
        for shift in (-1, 1):
            touches |= np.roll(outside, shift, axis=axis)
    assert (touches[removed]).all()
    assert interior_cells(body, 0).sum() == body.sum()


def test_a_bright_rim_does_not_enter_the_interior_sd():
    """Uniform tissue with a noisy outer layer: SD over the body > 0, interior SD = 0."""
    body = _ball()
    rim = body & ~interior_cells(body, 1)
    grid = np.where(body, 1.0, 0.0)
    grid[rim] = np.random.default_rng(0).uniform(0.0, 1.0, rim.sum())
    assert interior_sd(grid[None], body[None], 0)[0] > 0.1
    assert interior_sd(grid[None], body[None], 1)[0] == pytest.approx(0.0)


def test_too_small_body_is_refused():
    body = np.zeros((32, 32, 32), dtype=bool)
    body[10:12, 10:12, 10:12] = True  # 2³ cells: nothing is left after one layer
    with pytest.raises(ValueError):
        interior_sd(np.ones((1, 32, 32, 32)), body[None], 1)

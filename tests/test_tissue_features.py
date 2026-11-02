import numpy as np

from pearlatent.tissue_features import FEATURE_NAMES, tissue_features, tissue_masks


def _ball(size: int = 48, radius: float = 18.0, seed: int = 0) -> np.ndarray:
    """A bright ball (0.8, noise SD 0.02) in dark air (0.05) and a small enclosed cavity."""
    rng = np.random.default_rng(seed)
    z, y, x = np.meshgrid(*[np.arange(size)] * 3, indexing="ij")
    centre = (size - 1) / 2
    distance = np.sqrt((x - centre) ** 2 + (y - centre) ** 2 + (z - centre) ** 2)
    volume = np.full((size,) * 3, 0.05, dtype=np.float32)
    volume[distance <= radius] = 0.8
    volume[distance <= 3] = 0.05  # enclosed cavity at the centre
    volume += rng.normal(0, 0.02, volume.shape).astype(np.float32)
    return np.clip(volume, 0, 1)


def test_masks_split_the_fruit_into_tissue_and_air():
    m = tissue_masks(_ball(), window=11)
    assert (m["fruit"] == (m["tissue"] | m["air"])).all()
    assert not (m["tissue"] & m["air"]).any()
    assert (m["tissue"] == (m["low_density"] | m["high_density"])).all()
    assert not (m["low_density"] & m["high_density"]).any()
    # Cavity of radius 3 holds about 4/3 pi 27 = 113 voxels.
    assert 80 < m["air"].sum() < 160


def test_stray_object_is_not_tissue():
    volume = _ball()
    volume[0:3, 0:3, 0:3] = 0.8  # a bright object in a corner, not touching the ball
    m = tissue_masks(volume, window=11)
    assert not m["tissue"][0:3, 0:3, 0:3].any()


def test_a_dark_region_raises_the_spread_of_the_tissue():
    healthy = _ball()
    browned = healthy.copy()
    browned[24:32, 18:30, 18:30] -= (
        0.15  # darker tissue next to the core, still above air
    )
    a, b = tissue_features(healthy, window=11), tissue_features(browned, window=11)
    assert set(FEATURE_NAMES) <= set(a)
    assert b["tissue_sd"] > a["tissue_sd"]
    assert b["tissue_mean"] < a["tissue_mean"]
    assert 0 <= a["ks_low_high"] <= 1

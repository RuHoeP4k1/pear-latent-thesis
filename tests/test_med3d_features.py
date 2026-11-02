import numpy as np

from pearlatent.med3d_features import OBJECT_THRESHOLD, normalise_as_medicalnet


def test_object_voxels_get_mean_zero_and_sd_one():
    rng = np.random.default_rng(0)
    volume = np.full(
        (16, 16, 16), 0.02, dtype=np.float32
    )  # background below the threshold
    volume[4:12, 4:12, 4:12] = rng.uniform(0.5, 0.9, (8, 8, 8))
    out = normalise_as_medicalnet(volume)
    inside = volume > OBJECT_THRESHOLD
    assert abs(out[inside].mean()) < 1e-5
    assert abs(out[inside].std() - 1) < 1e-4
    # No noise: the same input gives the same output, and the background is 0 (nnU-Net).
    assert np.array_equal(out, normalise_as_medicalnet(volume))
    assert (out[~inside] == 0).all()


def test_brightness_and_contrast_of_the_fruit_are_removed():
    rng = np.random.default_rng(1)
    volume = np.zeros((16, 16, 16), dtype=np.float32)
    volume[4:12, 4:12, 4:12] = rng.uniform(0.4, 0.6, (8, 8, 8))
    brighter = volume.copy()
    brighter[volume > 0] = 0.2 + 1.5 * volume[volume > 0]
    inside = volume > OBJECT_THRESHOLD
    np.testing.assert_allclose(
        normalise_as_medicalnet(volume)[inside],
        normalise_as_medicalnet(brighter)[inside],
        atol=1e-4,
    )


def test_enclosed_cavity_stays_dark_and_outside_is_zero():
    volume = np.full((24, 24, 24), 0.02, dtype=np.float32)
    volume[4:20, 4:20, 4:20] = 0.8
    volume[10:14, 10:14, 10:14] = 0.02  # enclosed cavity, as dark as the air outside
    out = normalise_as_medicalnet(volume)
    assert (out[:4] == 0).all()  # outside the fruit
    assert (out[10:14, 10:14, 10:14] < -3).all()  # the cavity is not set to 0

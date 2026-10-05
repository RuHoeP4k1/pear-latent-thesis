import numpy as np
import polars as pl
import pytest

from pearlatent.exclusions import apply_exclusions, build_exclusions
from pearlatent.volume_qc import flag_volumes, volume_stats


def _cone(length: int = 126, shape=(128, 128, 128)) -> np.ndarray:
    """A cone along axis 2, wide at the high end, uint16 like the real files."""
    a = np.zeros(shape, dtype=np.uint16)
    zz, yy, xx = np.meshgrid(*(np.arange(n) for n in shape), indexing="ij")
    start = (shape[2] - length) // 2
    along = xx - start
    radius = 5 + 25 * along / length
    inside = (along >= 0) & (along < length)
    inside &= (zz - 64) ** 2 + (yy - 64) ** 2 <= radius**2
    a[inside] = 60000
    a[0, 0, 0] = 0
    return a


def test_volume_stats_extent_and_orientation():
    s = volume_stats(_cone(126))
    assert s["shape"] == "128x128x128"
    assert s["long_axis"] == 2
    assert s["long_extent"] == 126
    assert s["extent_axis0"] < 126
    # Widest slice at the high end of axis 2
    assert s["widest_slice_axis2"] > 100
    assert s["mean_in_fruit"] == pytest.approx(1.0)


def test_volume_stats_empty_and_constant_volumes():
    s = volume_stats(np.zeros((128, 128, 128), dtype=np.uint16))
    assert s["fruit_voxels"] == 0
    assert s["long_extent"] == 0


def _qc_row(fruit_id, shape="128x128x128", voxels=1000, long_extent=126, sha1=None):
    return {
        "fruit_id": fruit_id,
        "season": "2526",
        "shape": shape,
        "fruit_voxels": voxels,
        "long_extent": long_extent,
        "sha1": sha1 or fruit_id,
    }


def test_flag_volumes_reasons():
    qc = pl.DataFrame(
        [
            _qc_row("ok"),
            _qc_row("short", long_extent=100),
            _qc_row("small", shape="64x64x64"),
            _qc_row("empty", voxels=0, long_extent=0),
            _qc_row("dup1", sha1="same"),
            _qc_row("dup2", sha1="same"),
        ]
    )
    flags = dict(flag_volumes(qc).select("fruit_id", "qc_flag").iter_rows())
    assert flags["ok"] == ""
    assert "below 120" in flags["short"]
    assert "shape" in flags["small"]
    assert flags["empty"] == "empty volume"
    assert "identical" in flags["dup1"] and "identical" in flags["dup2"]


@pytest.fixture
def labels():
    return pl.DataFrame(
        {
            "fruit_id": ["2526_A01", "2526_A02", "2526_A03", "2425_A01", "2526_A04"],
            "season": ["2526", "2526", "2526", "2425", "2526"],
            "browning": [0, 3, 3, 3, 1],
            "cavity": [0, 3, 2, 3, 0],
        }
    )


@pytest.fixture
def volumes():
    # 2526_A04 has no volume
    return pl.DataFrame({"fruit_id": ["2526_A01", "2526_A02", "2526_A03", "2425_A01"]})


def test_build_exclusions_rules(labels, volumes):
    qc = pl.DataFrame(
        {
            "fruit_id": ["2526_A01", "2526_A03"],
            "season": ["2526", "2526"],
            "qc_flag": ["", "empty volume"],
        }
    )
    ex = build_exclusions(labels, volumes, qc)
    rows = {(r["fruit_id"], r["action"]) for r in ex.iter_rows(named=True)}
    assert rows == {
        ("2526_A04", "exclude"),  # unscanned
        ("2526_A02", "flag"),  # browning 3 and cavity 3
        ("2526_A03", "flag"),  # quality check
    }
    # The 2425 fruit with browning 3 and cavity 3 is not flagged: rule 10 concerns 2526
    assert "2425_A01" not in ex["fruit_id"].to_list()


def test_apply_exclusions_modes(labels, volumes):
    ex = build_exclusions(labels, volumes)
    primary = apply_exclusions(labels, ex, "primary")["fruit_id"].to_list()
    sensitivity = apply_exclusions(labels, ex, "sensitivity")["fruit_id"].to_list()
    assert "2526_A04" not in primary and "2526_A02" in primary
    assert "2526_A04" not in sensitivity and "2526_A02" not in sensitivity
    assert len(primary) == 4 and len(sensitivity) == 3
    with pytest.raises(ValueError, match="mode"):
        apply_exclusions(labels, ex, "all")

import numpy as np
import polars as pl
import pytest

from pearlatent.feature_sets import FEATURE_RUNS, load_features


class _Config:
    """Minimal stand-in for pearlatent.config.Config: only `path("derived_dir")`."""

    def __init__(self, root):
        self.root = root

    def path(self, key):
        assert key == "derived_dir"
        return self.root


def _fake_run(tmp_path):
    """A fake ct_32 run of 4 fruit (fruit k has all values k) and a sealed list with fruit 2."""
    folder, array_file = FEATURE_RUNS["ct_32"]
    run = tmp_path / folder
    run.mkdir(parents=True)
    ids = ["2526_A01", "2526_A02", "2526_F01", "2526_B01"]
    pl.DataFrame({"fruit_id": ids}).write_parquet(run / "manifest.parquet")
    np.save(
        run / array_file,
        np.arange(4, dtype=np.float32)[:, None, None, None] * np.ones((4, 2, 2, 2)),
    )
    pl.DataFrame(
        {
            "fruit_id": ["2526_F01"],
            "season": ["2526"],
            "sealed_part": ["2526 test boxes"],
        }
    ).write_csv(tmp_path / "sealed_fruit.csv")
    return _Config(tmp_path)


def test_rows_follow_the_requested_order(tmp_path):
    cfg = _fake_run(tmp_path)
    x = load_features(cfg, "ct_32", ["2526_B01", "2526_A01"])
    assert x.shape == (2, 8)
    assert x[0, 0] == 3 and x[1, 0] == 0


def test_sealed_fruit_refused_unless_asked(tmp_path):
    cfg = _fake_run(tmp_path)
    with pytest.raises(ValueError, match="not available"):
        load_features(cfg, "ct_32", ["2526_F01"])
    assert load_features(cfg, "ct_32", ["2526_F01"], include_sealed=True)[0, 0] == 2


def test_unknown_feature_set(tmp_path):
    with pytest.raises(ValueError, match="unknown"):
        load_features(_Config(tmp_path), "resnet", ["2526_A01"])

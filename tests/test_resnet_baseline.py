from pathlib import Path

import numpy as np
import polars as pl
import pytest

from pearlatent.resnet_baseline import check_fold_plan, fold_plan, safe_auc

BOXES = [f"2526_{c}" for c in "ABCDEGHIKLNO"]  # the 12 open boxes


def _folds() -> pl.DataFrame:
    """12 boxes in 4 outer folds of 3, as one repeat of outer_folds.csv."""
    return pl.DataFrame({"box": BOXES, "outer_fold": [1 + i // 3 for i in range(12)]})


def test_every_box_tested_once_with_three_roles():
    plan = fold_plan(_folds(), n_validation_boxes=2, seed=369)
    check_fold_plan(plan)
    per_fold = plan.group_by("outer_fold", "role").len().sort("outer_fold", "role")
    # Per fold: 3 test, 2 validation, 7 training boxes.
    assert set(zip(per_fold["role"], per_fold["len"], strict=True)) == {
        ("test", 3),
        ("validation", 2),
        ("train", 7),
    }
    tested = plan.filter(pl.col("role") == "test")["box"].sort().to_list()
    assert tested == sorted(BOXES)


def test_validation_boxes_never_test_boxes_of_same_fold():
    plan = fold_plan(_folds(), seed=369)
    for fold in range(1, 5):
        one = plan.filter(pl.col("outer_fold") == fold)
        validation = set(one.filter(pl.col("role") == "validation")["box"])
        test = set(_folds().filter(pl.col("outer_fold") == fold)["box"])
        assert not validation & test


def test_plan_does_not_depend_on_row_order():
    shuffled = _folds().sample(fraction=1.0, shuffle=True, seed=1)
    a = fold_plan(_folds(), seed=369).sort("outer_fold", "box")
    b = fold_plan(shuffled, seed=369).sort("outer_fold", "box")
    assert a.equals(b)


def test_check_fold_plan_rejects_a_box_tested_twice():
    plan = fold_plan(_folds(), seed=369)
    broken = plan.with_columns(
        role=pl.when((pl.col("outer_fold") == 2) & (pl.col("box") == "2526_A"))
        .then(pl.lit("test"))
        .otherwise(pl.col("role"))
    )
    with pytest.raises(ValueError):
        check_fold_plan(broken)


def test_safe_auc_one_class_is_nan():
    assert np.isnan(safe_auc(np.array([1, 1]), np.array([0.2, 0.9])))
    assert safe_auc(np.array([0, 1]), np.array([0.2, 0.9])) == 1.0


def _hugo_repo() -> Path:
    try:
        from pearlatent.config import load_config

        repo = load_config().path("hugo_repo")
    except Exception as err:
        pytest.skip(f"no config: {err}")
    if not (Path(repo) / "train_clf_3d.py").exists():
        pytest.skip(f"Hugo's clone not found at {repo}")
    return Path(repo)


def test_train_one_fold_on_small_random_volumes(tmp_path):
    """Hugo's ResNet3D trains and predicts on 16³ volumes; the log has one row per epoch."""
    torch = pytest.importorskip("torch")
    from pearlatent.resnet_baseline import import_hugo_resnet, run_epoch, train_one_fold

    ResNet3D = import_hugo_resnet(_hugo_repo())
    torch.manual_seed(0)

    class Tiny:
        def __init__(self, n):
            self.x = torch.rand(n, 1, 16, 16, 16)
            self.y = torch.tensor([i % 2 for i in range(n)])

        def __len__(self):
            return len(self.y)

        def __getitem__(self, i):
            return self.x[i], self.y[i]

    settings = {
        "base_channels": 4,
        "dropout": 0.0,
        "epochs": 3,
        "batch_size": 4,
        "lr": 1e-3,
        "weight_decay": 0.0,
        "early_stop_patience": 5,
        "min_epochs": 1,
        "amp": False,
    }
    log = tmp_path / "log.csv"
    model, best_epoch = train_one_fold(
        ResNet3D, {"train": Tiny(8), "validation": Tiny(4)}, settings, "cpu", 0, log
    )
    assert 2 <= best_epoch <= 3  # best is chosen only after min_epochs
    assert pl.read_csv(log).height == 3
    loader = torch.utils.data.DataLoader(Tiny(6), batch_size=4)
    out = run_epoch(model, loader, torch.nn.CrossEntropyLoss(), "cpu")
    assert out["probability"].shape == (6,)
    assert ((out["probability"] >= 0) & (out["probability"] <= 1)).all()

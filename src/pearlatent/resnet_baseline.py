"""ResNet-18 baseline trained from scratch on the 128³ CT, binary_23 (charter, Methods).

The one network trained from scratch in this thesis (Hugo, 13 October 2026). It shows what
supervised training on the same fruit reaches, beside the PLS-DA rows on frozen features.

Architecture: `ResNet3D` of Hugo's `train_clf_3d.py` (basic blocks 2-2-2-2, the ResNet-18
layout), imported from his clone at the pinned commit, not copied (no licence file). Training
settings are his defaults unless a comment below says otherwise.

Design (charter, Methods; `docs/cv_explained.md` section 5):

1. Fruit: the open 2526 fruit, faulty volumes dropped (315 on 27 November 2026). The sealed test
   boxes F, J, M are not in the encoding manifest, so they cannot enter here.
2. Outer folds: one partition only, repeat 1 of `derived/outer_folds_20/outer_folds.csv` (4 folds
   of 3 boxes), the same partition the PLS-DA rows use in repeat 1.
3. Inside each outer fold: `N_VALIDATION_BOXES` of the 9 training boxes are the validation group
   for early stopping, drawn with `SEED`. The model is not refitted on all 9 boxes afterwards
   (the number of epochs found on one validation group would not transfer to more data).
4. Output: for every fruit, the probability of binary_23 = 1 from the model of the fold in which
   its box was held out (out-of-fold), for the paired comparison with the PLS-DA rows.
5. Grades: the ordinal step of notebook 12 (`pearlatent.ordinal`) on that out-of-fold score,
   as for the PLS-DA rows (Ruben, 27 November 2026), on the probability and on its log-odds.
   Rerun on a copied run folder: `--ordinal-for results/<run>`. A network trained on the grades
   is a separate second run (charter, Methods).

Input: each volume min-max scaled to [0, 1], as Hugo's `utils/volumes.py` (`scale_minmax`).
No augmentation (Hugo's `--augment` needs torchio and is off by default) and no pseudo-labels
(his `--unlabeled_dir` is optional; we have no unlabeled fruit in this design).

Run: `uv run python -m pearlatent.resnet_baseline` (full run, GPU). Smoke run on a laptop:
`uv run python -m pearlatent.resnet_baseline --folds 1 --epochs 1 --max-fruit-per-box 3`.
"""

from __future__ import annotations

import argparse
import copy
import csv
import sys
from pathlib import Path

import numpy as np
import polars as pl

from pearlatent.encoding import PINNED_COMMIT, hugo_commit
from pearlatent.splits import assert_no_group_leakage

# The seed of this project (test-box draw, Hugo's VAE split); fixes the validation boxes,
# the weight initialisation and the batch order.
SEED = 369
# 2 of the 9 training boxes (about 50 fruit) for early stopping, 7 (about 185) for training.
# A choice: 3 boxes (one whole other outer fold) would leave only about 157 for training;
# 1 box (about 26 fruit) gives a validation loss too noisy to stop on.
N_VALIDATION_BOXES = 2
FOLD_REPEAT = 1  # the one partition the ResNet gets (docs/cv_explained.md section 5)

# Hugo's defaults in train_clf_3d.py (183112f).
HUGO_DEFAULTS = {
    "base_channels": 16,  # charter: 16 or 64 still to settle; 16 is his default
    "dropout": 0.2,
    "epochs": 100,
    "batch_size": 16,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "early_stop_patience": 20,
    # Hugo starts watching the validation loss only after epoch 10 (`pseudo_start_epoch`,
    # also without pseudo-labels). Kept, so the best model is never one of the first 10 epochs.
    "min_epochs": 10,
}


# ---------------------------------------------------------------------------
# Folds
# ---------------------------------------------------------------------------


def read_outer_folds(path: Path, repeat: int = FOLD_REPEAT) -> pl.DataFrame:
    """Columns `box`, `outer_fold` for one repeat of the outer-fold file."""
    folds = pl.read_csv(path).filter(pl.col("repeat") == repeat)
    if folds.height == 0:
        raise ValueError(f"repeat {repeat} not in {path}")
    return folds.select("box", pl.col("fold").alias("outer_fold"))


def fold_plan(
    boxes_and_folds: pl.DataFrame,
    n_validation_boxes: int = N_VALIDATION_BOXES,
    seed: int = SEED,
) -> pl.DataFrame:
    """Which boxes train, validate and test in each outer fold.

    In: one row per box with `box` and `outer_fold`. Out: one row per (outer_fold, box) with
    `role` = "test" (the boxes of that fold), "validation" (`n_validation_boxes` drawn from the
    other boxes) or "train" (the rest). The draw is per fold from one generator seeded with
    `seed`, over the sorted box names, so it does not depend on row order.
    """
    rng = np.random.default_rng(seed)
    all_boxes = sorted(boxes_and_folds["box"].to_list())
    rows = []
    for fold in sorted(boxes_and_folds["outer_fold"].unique().to_list()):
        test = set(
            boxes_and_folds.filter(pl.col("outer_fold") == fold)["box"].to_list()
        )
        rest = [b for b in all_boxes if b not in test]
        validation = set(
            rng.choice(rest, size=n_validation_boxes, replace=False).tolist()
        )
        for box in all_boxes:
            role = (
                "test"
                if box in test
                else "validation"
                if box in validation
                else "train"
            )
            rows.append({"outer_fold": fold, "box": box, "role": role})
    return pl.DataFrame(rows)


def check_fold_plan(plan: pl.DataFrame) -> None:
    """Raise unless every box is tested exactly once and the three roles never share a box."""
    tested = plan.filter(pl.col("role") == "test")["box"]
    if tested.n_unique() != plan["box"].n_unique() or tested.len() != tested.n_unique():
        raise ValueError("every box must be in the test role exactly once")
    for fold in plan["outer_fold"].unique().to_list():
        one = plan.filter(pl.col("outer_fold") == fold)
        # Within one fold each box has one role: the repository's leakage check, box by role.
        assert_no_group_leakage(one, group_col="box", split_col="role")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def resnet_fruit(cfg, max_fruit_per_box: int = 0) -> pl.DataFrame:
    """The fruit table for the baseline: open 2526 fruit without faulty volumes, with `binary_23`.

    `max_fruit_per_box` > 0 keeps the first fruit of each box (by `fruit_id`), only for a smoke
    run. Prints how many fruit go in and come out of every filter.
    """
    from pearlatent.fruit_table import drop_faulty_volumes, open_fruit

    fruit = open_fruit(cfg)
    print(f"open fruit: {fruit.height}")
    fruit = fruit.filter(pl.col("season") == "2526")
    print(f"2526 only: {fruit.height}")
    fruit = drop_faulty_volumes(fruit, cfg.path("derived_dir") / "volume_decisions.csv")
    print(f"without faulty volumes: {fruit.height}")
    fruit = fruit.with_columns(
        binary_23=(pl.col("defective") >= 2).cast(pl.Int64)
    ).sort("fruit_id")
    if max_fruit_per_box > 0:
        fruit = (
            fruit.group_by("box", maintain_order=True)
            .head(max_fruit_per_box)
            .sort("fruit_id")
        )
        print(f"smoke run, at most {max_fruit_per_box} per box: {fruit.height}")
    return fruit.select(
        "fruit_id",
        "box",
        "file_stem",
        "browning",
        "cavity",
        "defective",
        "cause",
        "binary_23",
    )


class VolumeDataset:
    """(volume, label) pairs for a torch DataLoader: volume (1, 128, 128, 128) float32 in [0, 1].

    Loads each NIfTI file from disk when asked (8 MB per volume as float32), so memory stays
    small; on the workstation the operating system's file cache makes later epochs fast.
    """

    def __init__(self, paths: list[Path], labels: list[int]):
        if len(paths) != len(labels):
            raise ValueError("one label per volume")
        self.paths = list(paths)
        self.labels = list(labels)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        import nibabel as nib
        import torch

        from pearlatent.pooling import scale_minmax

        volume = scale_minmax(np.asarray(nib.load(self.paths[index]).dataobj))
        x = torch.from_numpy(np.ascontiguousarray(volume))[None]  # add the channel axis
        return x, torch.tensor(self.labels[index], dtype=torch.long)


# ---------------------------------------------------------------------------
# Model and training
# ---------------------------------------------------------------------------


def import_hugo_resnet(repo: str | Path):
    """Hugo's `ResNet3D` class from `train_clf_3d.py`, from a clone at the pinned commit."""
    repo = Path(repo)
    commit = hugo_commit(repo)
    if not commit.startswith(PINNED_COMMIT):
        raise RuntimeError(
            f"{repo} is at commit {commit[:7]}, expected {PINNED_COMMIT}"
        )
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from train_clf_3d import ResNet3D

    return ResNet3D


def run_epoch(
    model, loader, criterion, device: str, optimizer=None, amp: bool = False
) -> dict:
    """One pass over `loader`; trains when `optimizer` is given, otherwise only evaluates.

    Out: mean loss over the volumes, and the probability of class 1 and the label of every
    volume in loader order (the loader of an evaluation pass must not shuffle).
    """
    import torch

    training = optimizer is not None
    model.train(training)
    use_amp = amp and device == "cuda"
    total_loss, n = 0.0, 0
    probabilities, labels = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(training):
            with torch.autocast("cuda", enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * len(y)
        n += len(y)
        probabilities.append(
            torch.softmax(logits.float(), dim=1)[:, 1].detach().cpu().numpy()
        )
        labels.append(y.cpu().numpy())
    return {
        "loss": total_loss / max(n, 1),
        "probability": np.concatenate(probabilities) if probabilities else np.array([]),
        "label": np.concatenate(labels) if labels else np.array([]),
    }


def safe_auc(label: np.ndarray, score: np.ndarray) -> float:
    """ROC AUC, or NaN when only one class is present (possible in a smoke run)."""
    from sklearn.metrics import roc_auc_score

    if len(np.unique(label)) < 2:
        return float("nan")
    return float(roc_auc_score(label, score))


def train_one_fold(
    ResNet3D,
    data: dict[str, VolumeDataset],
    settings: dict,
    device: str,
    num_workers: int,
    log_path: Path,
) -> tuple[object, int]:
    """Train on `data["train"]`, stop on the loss of `data["validation"]`; return the best model.

    Best model = lowest validation loss after `min_epochs`; training stops after
    `early_stop_patience` epochs without a lower validation loss (Hugo's rule). Every epoch is
    written to `log_path` (CSV). Out: the model with the best weights loaded, and its epoch.
    """
    import torch
    from torch.utils.data import DataLoader

    model = ResNet3D(
        output_size=2,
        base_channels=settings["base_channels"],
        dropout=settings["dropout"],
    )
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=settings["epochs"]
    )

    pin = device == "cuda"
    train_loader = DataLoader(
        data["train"],
        batch_size=settings["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
    )
    validation_loader = DataLoader(
        data["validation"],
        batch_size=settings["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )

    best_loss, best_epoch, best_state, waited = np.inf, 0, None, 0
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch", "train_loss", "validation_loss", "validation_auc", "lr"]
        )
    for epoch in range(1, settings["epochs"] + 1):
        train = run_epoch(
            model, train_loader, criterion, device, optimizer, settings["amp"]
        )
        validation = run_epoch(
            model, validation_loader, criterion, device, None, settings["amp"]
        )
        lr = scheduler.get_last_lr()[0]
        scheduler.step()
        auc = safe_auc(validation["label"], validation["probability"])
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, train["loss"], validation["loss"], auc, lr])
        print(
            f"  epoch {epoch}: train loss {train['loss']:.4f}, validation loss "
            f"{validation['loss']:.4f}, validation AUC {auc:.3f}",
            flush=True,
        )

        if epoch <= settings["min_epochs"]:
            continue
        if validation["loss"] < best_loss:
            best_loss, best_epoch, waited = validation["loss"], epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            waited += 1
            if waited >= settings["early_stop_patience"]:
                print(f"  early stop at epoch {epoch}; best epoch {best_epoch}")
                break
    if best_state is None:
        # The run ended before `min_epochs` (a smoke run): keep the last epoch.
        return model, epoch
    model.load_state_dict(best_state)
    return model, best_epoch


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Ordinal step on the out-of-fold score (as notebook 12 for the PLS-DA rows)
# ---------------------------------------------------------------------------

GRADES = (0, 1, 2, 3)
# The ordinal model gets the score on two scales, to compare before one is chosen (Ruben):
# "probability" as saved, and "log_odds" = log(p / (1 - p)). The PLS-DA prediction that
# notebook 12 uses is roughly linear in the grade; a network probability is squeezed towards 0
# and 1, and the cumulative logit model is linear in its score, so log-odds may fit better.
SCORE_SCALES = ("probability", "log_odds")
# Probabilities are clipped to [EPS, 1 - EPS] before the log-odds, so 0 or 1 gives no infinity.
EPS = 1e-6


def score_on_scale(probability: np.ndarray, scale: str) -> np.ndarray:
    """The out-of-fold probability as the ordinal model's score: as it is, or its log-odds."""
    p = np.asarray(probability, dtype=float)
    if scale == "probability":
        return p
    if scale == "log_odds":
        p = np.clip(p, EPS, 1 - EPS)
        return np.log(p / (1 - p))
    raise ValueError(f"unknown scale {scale}")


def ordinal_step(oof: pl.DataFrame, plan: pl.DataFrame, scale: str) -> pl.DataFrame:
    """Grade probabilities from the ResNet's out-of-fold score, by the method of notebook 12.

    In: the out-of-fold table (`fruit_id`, `box`, `defective`, `probability`), the fold plan
    (`outer_fold`, `box`, `role`) and the score scale. For every outer fold the cumulative
    logit model (`pearlatent.ordinal`) is fitted on the out-of-fold scores of the other three
    folds and predicts this fold. One partition, so there is no average over repeats.
    Out: `oof` with `scale`, `p_grade_0` .. `p_grade_3`, `assigned` and `expected_grade`.

    Same limit as notebook 12: the score of a training fruit comes from a network fitted
    partly on the held-out boxes; the ordinal step has four parameters, so the effect is small.
    """
    from pearlatent.ordinal import cross_validated_ordinal

    folds = plan.filter(pl.col("role") == "test").select(
        pl.lit(FOLD_REPEAT).alias("repeat"), pl.col("outer_fold").alias("fold"), "box"
    )
    p = cross_validated_ordinal(
        score_on_scale(oof["probability"].to_numpy(), scale),
        oof["defective"].to_numpy(),
        oof["box"].to_numpy(),
        folds,
        GRADES,
    )
    return oof.with_columns(
        scale=pl.lit(scale),
        **{f"p_grade_{k}": pl.Series(p[:, i]) for i, k in enumerate(GRADES)},
        assigned=pl.Series(np.array(GRADES)[p.argmax(axis=1)]),
        expected_grade=pl.Series(p @ np.array(GRADES, dtype=float)),
    )


def ordinal_for_run(run_dir: Path) -> pl.DataFrame:
    """The ordinal step on both scales for a finished run folder; writes two files there.

    `ordinal_predictions.parquet`: one row per fruit and scale. `ordinal_summary.csv`: the grade
    metrics of notebook 12 (`pearlatent.ordinal.grade_summary`) per scale, all fruit and split by
    cause (charter rule 5). Out: the summary table.
    """
    from pearlatent.ordinal import grade_summary

    run_dir = Path(run_dir)
    oof = pl.read_parquet(run_dir / "oof_predictions.parquet")
    plan = pl.read_csv(run_dir / "fold_plan.csv")
    tables = [ordinal_step(oof, plan, scale) for scale in SCORE_SCALES]
    predictions = pl.concat(tables)
    predictions.write_parquet(run_dir / "ordinal_predictions.parquet")

    rows = []
    for table in tables:
        # Grade 0 fruit have cause "none"; every defect cause is reported with them beside it.
        groups = [("all", table)] + [
            (c, table.filter(pl.col("cause").is_in(["none", c])))
            for c in ("browning only", "cavity only", "both")
        ]
        for name, part in groups:
            probabilities = part.select([f"p_grade_{k}" for k in GRADES]).to_numpy()
            # A group may lack a grade (cavity only has almost no grade 3): recall is then NaN.
            with np.errstate(invalid="ignore", divide="ignore"):
                row = grade_summary(part["defective"].to_numpy(), probabilities, GRADES)
            rows.append({"scale": table["scale"][0], "cause": name, **row})
    summary = pl.DataFrame(rows)
    summary.write_csv(run_dir / "ordinal_summary.csv")
    with pl.Config(tbl_cols=-1, tbl_width_chars=200, float_precision=3):
        print(summary.filter(pl.col("cause") == "all"))
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--ordinal-for",
        type=str,
        default="",
        help="only run the ordinal step on this finished run folder",
    )
    parser.add_argument(
        "--folds",
        type=str,
        default="",
        help="outer folds to run, e.g. '1' or '1,3'; empty = all",
    )
    parser.add_argument(
        "--max-fruit-per-box", type=int, default=0, help="smoke run only; 0 = all fruit"
    )
    parser.add_argument(
        "--num-workers", type=int, default=None, help="default: num_workers in config"
    )
    parser.add_argument(
        "--device", type=str, default=None, help="default: device in config"
    )
    parser.add_argument("--amp", action="store_true", help="mixed precision on CUDA")
    for name, value in HUGO_DEFAULTS.items():
        parser.add_argument(
            f"--{name.replace('_', '-')}", type=type(value), default=value
        )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    import torch

    from pearlatent.config import load_config
    from pearlatent.device import get_device, seed_everything
    from pearlatent.runs import new_run_dir, write_run_record

    args = parse_args(argv)
    if args.ordinal_for:
        ordinal_for_run(Path(args.ordinal_for))
        return
    settings = {name: getattr(args, name) for name in HUGO_DEFAULTS} | {"amp": args.amp}
    cfg = load_config()
    device = get_device(args.device or cfg.compute("device", "auto"))
    num_workers = (
        args.num_workers
        if args.num_workers is not None
        else int(cfg.compute("num_workers", 0))
    )
    ResNet3D = import_hugo_resnet(cfg.path("hugo_repo"))

    fruit = resnet_fruit(cfg, args.max_fruit_per_box)
    plan = fold_plan(
        read_outer_folds(cfg.path("derived_dir") / "outer_folds_20" / "outer_folds.csv")
    )
    check_fold_plan(plan)
    missing = set(fruit["box"].unique().to_list()) - set(plan["box"].to_list())
    if missing:
        raise ValueError(f"boxes without an outer fold: {sorted(missing)}")
    folds = (
        [int(k) for k in args.folds.split(",")]
        if args.folds
        else sorted(plan["outer_fold"].unique().to_list())
    )

    run_name = "resnet18_binary23" + ("_smoke" if args.max_fruit_per_box else "")
    run_dir = new_run_dir(cfg.path("results_dir"), run_name)
    write_run_record(
        run_dir,
        {
            "seed": SEED,
            "fold_repeat": FOLD_REPEAT,
            "n_validation_boxes": N_VALIDATION_BOXES,
            "folds": folds,
            "max_fruit_per_box": args.max_fruit_per_box,
            "fruit": fruit.height,
            "device": device,
            "num_workers": num_workers,
            "hugo_commit": PINNED_COMMIT,
            "architecture": "train_clf_3d.ResNet3D, blocks 2-2-2-2, output 2",
            **settings,
        },
        root=cfg.root,
    )
    plan.write_csv(run_dir / "fold_plan.csv")
    print(f"device {device}; run folder {run_dir}")

    def dataset(rows: pl.DataFrame) -> VolumeDataset:
        paths = [cfg.path("ct_dir_2526") / f"{stem}.nii" for stem in rows["file_stem"]]
        return VolumeDataset(paths, rows["binary_23"].to_list())

    predictions = []
    for fold in folds:
        roles = plan.filter(pl.col("outer_fold") == fold).select("box", "role")
        part = fruit.join(roles, on="box", how="inner", validate="m:1")
        assert_no_group_leakage(part, group_col="box", split_col="role")
        counts = {
            r: part.filter(pl.col("role") == r).height
            for r in ("train", "validation", "test")
        }
        print(
            f"outer fold {fold}: {counts} fruit; test boxes "
            f"{sorted(part.filter(pl.col('role') == 'test')['box'].unique().to_list())}"
        )
        # Same seed in every fold, so a fold's result does not depend on which folds ran before.
        seed_everything(SEED)
        data = {
            r: dataset(part.filter(pl.col("role") == r))
            for r in ("train", "validation")
        }
        model, best_epoch = train_one_fold(
            ResNet3D,
            data,
            settings,
            device,
            num_workers,
            run_dir / f"log_fold{fold}.csv",
        )
        torch.save(model.state_dict(), run_dir / f"best_fold{fold}.pth")

        test_rows = part.filter(pl.col("role") == "test")
        loader = torch.utils.data.DataLoader(
            dataset(test_rows),
            batch_size=settings["batch_size"],
            shuffle=False,
            num_workers=num_workers,
        )
        test = run_epoch(model, loader, torch.nn.CrossEntropyLoss(), device)
        predictions.append(
            test_rows.with_columns(
                outer_fold=pl.lit(fold),
                best_epoch=pl.lit(best_epoch),
                probability=pl.Series(test["probability"]),
            ).drop("file_stem")
        )
        print(
            f"outer fold {fold}: best epoch {best_epoch}, test AUC "
            f"{safe_auc(test['label'], test['probability']):.3f} on {test_rows.height} fruit"
        )

    out = pl.concat(predictions)
    out.write_parquet(run_dir / "oof_predictions.parquet")
    print(
        f"out-of-fold predictions for {out.height} fruit; pooled AUC "
        f"{safe_auc(out['binary_23'].to_numpy(), out['probability'].to_numpy()):.3f}"
    )
    if sorted(folds) == sorted(plan["outer_fold"].unique().to_list()):
        ordinal_for_run(run_dir)
    else:
        print("ordinal step skipped: it needs the out-of-fold score of every fold")
    print(f"wrote {run_dir}")


if __name__ == "__main__":
    main()

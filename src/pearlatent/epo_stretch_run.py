"""PLS-DA on `binary_23` with EPO on the measured stretch factor, latent rows (27 November 2026).

Ruben, 27 November 2026: the latent scores follow the stalk / stretch at equal real size
(`docs/methods_plsda.md` sections 21 and 22), the other rows do not; the stretch is an
artefact of Hugo's rescaling to 128 voxels, real size is biology and stays. So the stretch
direction (`pearlatent.epo.fit_stretch_direction`) is removed from the three latent rows,
inside every fold, and nothing else changes: the same fruit (315), folds (20 repeats of the
4-fold box cross-validation) and centring only. LVs: `FINAL_N_LV` (Ruben, 27 November 2026,
from the curves with EPO); the first runs of 27 November used the LVs of the rows without EPO. The full error curve is saved, so a shift of
the elbow can be seen.

Per row, written to `results/<run_id>_plsda_binary23_<row>_epo_stretch/`:
`predictions.parquet` (the same columns as the saved rows, so `pearlatent.stretch_check`
and the comparison notebooks read it), `curves.parquet`, `scores.json`, `run.json`. Printed:
AUC with and without, the paired difference, and the healthy-fruit partial correlations with
the stretch factor, real volume and stalk length, before and after.

    uv run python -m pearlatent.epo_stretch_run
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl

LATENT_ROWS = ("hugo_vae", "hugo_vae_hr", "hugo_vae_hr_sd")
# LVs of the final latent rows with EPO: Ruben, 27 November 2026, by eye at the elbow of the
# mean RMSECV curves with EPO (notebook 23, figure 7), before any test box is opened.
# None = the LVs chosen for the row without EPO (the first runs of 27 November).
FINAL_N_LV = {"hugo_vae": 3, "hugo_vae_hr": 2, "hugo_vae_hr_sd": 2}


def model_fruit(cfg) -> pl.DataFrame:
    """The 315 open 2526 fruit of the saved rows (notebook 08), with stretch and real size."""
    from pearlatent.fruit_table import drop_faulty_volumes, open_fruit
    from pearlatent.runs import REAL_SIZE_RUN_DIR

    open_2526 = open_fruit(cfg).filter(pl.col("season") == "2526")
    fruit = drop_faulty_volumes(
        open_2526, cfg.path("derived_dir") / "volume_decisions.csv"
    ).with_columns(binary_23=(pl.col("defective") >= 2).cast(pl.Int8))
    print(f"open 2526 fruit {open_2526.height}, after faulty volumes {fruit.height}")
    sizes = pl.read_parquet(
        cfg.path("derived_dir") / REAL_SIZE_RUN_DIR / "real_size.parquet"
    ).select(
        "fruit_id",
        "stretch_factor",
        "fruit_length_mm",
        "body_volume_cm3",
        "stalk_length_mm",
    )
    joined = fruit.join(sizes, on="fruit_id", how="left")
    if joined["stretch_factor"].null_count():
        raise ValueError("some fruit have no stretch factor")
    return joined


def healthy_links(table: pl.DataFrame) -> dict:
    """Partial Spearman on the healthy fruit, score minus box median (check C style).

    Stretch given real volume, real volume given stretch, stalk length given real volume.
    """
    from pearlatent.nuisance import partial_spearman

    healthy = table.filter(pl.col("defective") == 0).with_columns(
        score_in_box=pl.col("score") - pl.col("score").median().over("box")
    )
    score = healthy["score_in_box"].to_numpy()
    stretch = healthy["stretch_factor"].to_numpy()
    volume = healthy["body_volume_cm3"].to_numpy()
    stalk = healthy["stalk_length_mm"].to_numpy()
    return {
        "stretch_given_volume": partial_spearman(score, stretch, volume),
        "volume_given_stretch": partial_spearman(score, volume, stretch),
        "stalk_given_volume": partial_spearman(score, stalk, volume),
    }


def main() -> None:
    """Cross-validate the three latent rows with stretch EPO and save one run per row."""
    from pearlatent.config import load_config
    from pearlatent.feature_sets import load_features
    from pearlatent.nuisance import box_bootstrap
    from pearlatent.plsda import (
        CUTOFF,
        MAX_COMPONENTS,
        box_bootstrap_auc,
        classification_scores,
        cross_validated_predictions,
        mean_squared_errors,
        paired_box_bootstrap_auc_difference,
        youden_cutoff,
    )
    from pearlatent.runs import PLSDA_BINARY23_RUN_DIRS, new_run_dir, write_run_record

    cfg = load_config()
    results = cfg.path("results_dir")
    fruit = model_fruit(cfg)
    fruit_ids = fruit["fruit_id"].to_list()
    y = fruit["binary_23"].to_numpy()
    grade = fruit["defective"].to_numpy()
    boxes = fruit["box"].to_numpy()
    folds = pl.read_csv(cfg.path("derived_dir") / "outer_folds_20" / "outer_folds.csv")
    epo_stretch = (
        (fruit["defective"] == 0).to_numpy(),
        fruit["stretch_factor"].to_numpy(),
        fruit["body_volume_cm3"].to_numpy(),
    )

    for row in LATENT_ROWS:
        base_dir = results / PLSDA_BINARY23_RUN_DIRS[row]
        base_n_lv = json.loads((base_dir / "scores.json").read_text())["n_lv"]
        n_lv = FINAL_N_LV.get(row) or base_n_lv
        base = pl.read_parquet(base_dir / "predictions.parquet")
        if base["fruit_id"].to_list() != fruit_ids:
            raise ValueError(f"{row}: fruit order differs from the saved run")

        x = load_features(cfg, row, fruit_ids).astype(float)
        n_components = min(MAX_COMPONENTS, x.shape[1])
        predictions = cross_validated_predictions(
            x,
            y,
            boxes,
            folds,
            scale=False,
            max_components=n_components,
            epo_stretch=epo_stretch,
        )
        rmsecv = np.sqrt(mean_squared_errors(predictions, y).mean(axis=0))
        curves = pl.DataFrame(
            {"n_lv": np.arange(1, n_components + 1), "rmsecv": rmsecv}
        )
        # Average of the 20 out-of-fold predictions at the row's chosen LVs (as notebook 08).
        mean_prediction = predictions[:, :, n_lv - 1].mean(axis=0)
        cutoff = youden_cutoff(mean_prediction, y)
        scores = classification_scores(mean_prediction, y, grade, cutoff=CUTOFF)
        scores["youden"] = {
            "cutoff": cutoff,
            **classification_scores(mean_prediction, y, grade, cutoff=cutoff),
        }
        scores["n_lv"] = n_lv
        low, high, _ = box_bootstrap_auc(mean_prediction, y, boxes)
        scores["auc_box_bootstrap_95"] = [low, high]
        paired = paired_box_bootstrap_auc_difference(
            mean_prediction, base["prediction"].to_numpy(), y, boxes
        )
        scores["paired_minus_without_epo"] = {
            k: paired[k] for k in ("difference", "lower_95", "upper_95")
        }

        table = base.with_columns(
            prediction=pl.Series(mean_prediction),
            called_positive=pl.Series(mean_prediction >= CUTOFF),
            called_positive_youden=pl.Series(mean_prediction >= cutoff),
        )
        sized = fruit.select(
            "fruit_id", "stretch_factor", "body_volume_cm3", "stalk_length_mm"
        )
        links = {}
        for name, t in (("without", base), ("with", table)):
            t = t.rename({"prediction": "score"}).join(sized, on="fruit_id")
            links[name] = healthy_links(t)
            for key in list(links[name]):
                low, high = box_bootstrap(lambda d, k=key: healthy_links(d)[k], t)
                links[name][f"{key}_low"], links[name][f"{key}_high"] = low, high
        scores["healthy_links"] = links

        out_dir = new_run_dir(results, f"plsda_binary23_{row}_epo_stretch")
        table.write_parquet(out_dir / "predictions.parquet")
        curves.write_parquet(out_dir / "curves.parquet")
        (out_dir / "scores.json").write_text(json.dumps(scores, indent=2))
        write_run_record(
            out_dir,
            {
                "what": "PLS-DA on binary_23 with EPO on the stretch factor, design A",
                "representation": row,
                "base_run": PLSDA_BINARY23_RUN_DIRS[row],
                "n_fruit": table.height,
                "n_lv": n_lv,
                "n_lv_rule": "Ruben, 27 November 2026, elbow of the RMSECV curve with EPO",
                "base_n_lv": base_n_lv,
                "epo": "stretch direction at equal real volume, healthy training fruit, per fold",
                "scale": False,
                "cutoff": CUTOFF,
                "cutoff_youden": cutoff,
                "folds": "derived/outer_folds_20/outer_folds.csv",
                "bootstrap_seed": 369,
            },
            root=cfg.root,
        )

        print(f"\n{row} ({n_lv} LVs) -> {out_dir.name}")
        print(
            f"  AUC {paired['auc_b']:.3f} -> {paired['auc_a']:.3f}, paired "
            f"{paired['difference']:+.3f} ({paired['lower_95']:+.3f} to "
            f"{paired['upper_95']:+.3f})"
        )
        best = int(np.argmin(rmsecv)) + 1
        print(
            "  RMSECV with EPO, LV 1 to 8: "
            + " ".join(f"{v:.3f}" for v in rmsecv[:8])
            + f" (lowest at {best})"
        )
        for key in (
            "stretch_given_volume",
            "volume_given_stretch",
            "stalk_given_volume",
        ):
            w, e = links["without"], links["with"]
            print(
                f"  {key:22s} {w[key]:+.2f} ({w[key + '_low']:+.2f} to "
                f"{w[key + '_high']:+.2f}) -> {e[key]:+.2f} ({e[key + '_low']:+.2f} to "
                f"{e[key + '_high']:+.2f})"
            )


if __name__ == "__main__":
    main()

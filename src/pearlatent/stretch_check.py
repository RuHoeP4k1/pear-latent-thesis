"""Which model rows does the stretch to 128 voxels reach? (27 November 2026, Ruben.)

Every row gets the same rescaled volumes, but not every row's score need follow the stretch
factor. This run answers it per row from the saved out-of-fold scores; no model is fitted.

1. Per row (the `binary_23` PLS-DA rows of `PLSDA_BINARY23_RUN_DIRS`), on the healthy fruit:
   `healthy_stretch_correlations`, with 95 % box-bootstrap intervals on the two partial
   correlations. A row whose partial with the stretch factor excludes 0 gets the correction
   (EPO on the stretch factor); a row whose partial with real volume excludes 0 follows the
   real size of the fruit.
2. Without any model, on all open fruit: the AUC of real body volume alone and of the
   stretch factor alone for `binary_23` (is a larger fruit more often non-consumable?).

Run (writes `results/<run_id>/stretch_check.parquet`, `size_auc.parquet`, a figure and
`run.json`):

    uv run python -m pearlatent.stretch_check
"""

from __future__ import annotations

import polars as pl

# The row drawn in the figure: the headline latent row (methods section 9).
FIGURE_ROW = "hugo_vae_hr_sd"


def size_auc(table: pl.DataFrame, column: str) -> float:
    """AUC of one size measure alone for `binary_23` (larger value = predicted defective)."""
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(table["binary_23"], table[column]))


def draw_figure(table: pl.DataFrame, row: str, path) -> None:
    """Healthy fruit: score (minus box median) against real body volume, coloured by s."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    healthy = table.filter(pl.col("defective") == 0).with_columns(
        score_in_box=pl.col("score") - pl.col("score").median().over("box")
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    points = axes[0].scatter(
        healthy["body_volume_cm3"],
        healthy["score_in_box"],
        c=healthy["stretch_factor"],
        cmap="viridis",
        s=18,
    )
    axes[0].set_xlabel("real body volume (cm³)")
    axes[0].set_ylabel("score minus box median")
    fig.colorbar(points, ax=axes[0], label="stretch factor s")
    points = axes[1].scatter(
        healthy["stretch_factor"],
        healthy["score_in_box"],
        c=healthy["body_volume_cm3"],
        cmap="magma",
        s=18,
    )
    axes[1].set_xlabel("stretch factor s = 128 / original length")
    fig.colorbar(points, ax=axes[1], label="real body volume (cm³)")
    fig.suptitle(
        f"{row}: {healthy.height} healthy fruit (defective 0), out-of-fold score"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    """Check per row and the model-free size AUC; write the run."""
    from pearlatent.config import load_config
    from pearlatent.nuisance import box_bootstrap, healthy_stretch_correlations
    from pearlatent.runs import (
        PLSDA_BINARY23_RUN_DIRS,
        REAL_SIZE_RUN_DIR,
        new_run_dir,
        write_run_record,
    )

    cfg = load_config()
    results = cfg.path("results_dir")
    sizes = pl.read_parquet(
        cfg.path("derived_dir") / REAL_SIZE_RUN_DIR / "real_size.parquet"
    ).select("fruit_id", "stretch_factor", "body_volume_cm3")

    rows = []
    tables = {}
    for row, run in PLSDA_BINARY23_RUN_DIRS.items():
        predictions = pl.read_parquet(results / run / "predictions.parquet")
        table = predictions.rename({"prediction": "score"}).join(
            sizes, on="fruit_id", how="inner"
        )
        if table.height != predictions.height:
            raise ValueError(
                f"{row}: {predictions.height - table.height} fruit without size"
            )
        tables[row] = table
        stats = healthy_stretch_correlations(table)
        # Bootstrap over whole boxes (charter rule 4), the same draws for both partials.
        for key in ("partial_stretch_given_volume", "partial_volume_given_stretch"):
            low, high = box_bootstrap(
                lambda t, key=key: healthy_stretch_correlations(t)[key], table
            )
            stats[f"{key}_low"], stats[f"{key}_high"] = low, high
        rows.append({"row": row, "fruit": table.height, **stats})
        print(
            f"{row:16s} n_healthy {stats['n_healthy']}: "
            f"stretch | volume {stats['partial_stretch_given_volume']:+.2f} "
            f"({stats['partial_stretch_given_volume_low']:+.2f} to "
            f"{stats['partial_stretch_given_volume_high']:+.2f}); "
            f"volume | stretch {stats['partial_volume_given_stretch']:+.2f} "
            f"({stats['partial_volume_given_stretch_low']:+.2f} to "
            f"{stats['partial_volume_given_stretch_high']:+.2f})"
        )
    check = pl.DataFrame(rows)

    # Model-free: the fruit of any row (same 315 open fruit in every row).
    fruit = tables[FIGURE_ROW]
    auc_rows = []
    for column in ("body_volume_cm3", "stretch_factor"):
        low, high = box_bootstrap(lambda t, c=column: size_auc(t, c), fruit)
        auc_rows.append(
            {
                "measure": column,
                "auc": size_auc(fruit, column),
                "low": low,
                "high": high,
            }
        )
        print(
            f"{column} alone for binary_23: AUC {auc_rows[-1]['auc']:.3f} ({low:.3f} to {high:.3f})"
        )
    auc_table = pl.DataFrame(auc_rows)

    out_dir = new_run_dir(results, "stretch_check")
    check.write_parquet(out_dir / "stretch_check.parquet")
    auc_table.write_parquet(out_dir / "size_auc.parquet")
    draw_figure(
        tables[FIGURE_ROW], FIGURE_ROW, out_dir / f"{FIGURE_ROW}_healthy_stretch.png"
    )
    write_run_record(
        out_dir,
        {
            "real_size_run": REAL_SIZE_RUN_DIR,
            "rows": PLSDA_BINARY23_RUN_DIRS,
            "fruit": fruit.height,
            "figure_row": FIGURE_ROW,
        },
    )
    print(f"written: {out_dir}")


if __name__ == "__main__":
    main()

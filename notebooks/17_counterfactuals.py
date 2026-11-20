import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="17 Counterfactual images")

with app.setup(hide_code=True):
    import json

    import marimo as mo
    import matplotlib.pyplot as plt
    import nibabel as nib
    import numpy as np
    import polars as pl

    from pearlatent.config import load_config
    from pearlatent.counterfactual import DRAW_SEED, draw_example_fruit
    from pearlatent.fruit_table import drop_faulty_volumes, open_fruit
    from pearlatent.pooling import scale_minmax
    from pearlatent.reconstruction import body_centre, body_voxel_mask, central_planes
    from pearlatent.runs import (
        ENCODING_RUN_DIR,
        PLSDA_BINARY23_RUN_DIRS,
        latest_run_dir,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 17 Counterfactual images

    **Question.** What does a fruit look like when its latent `mu` is moved along the direction
    that a PLS-DA model uses to call a fruit non-consumable? Charter, Extensions.

    **Plan** (Ruben, 7 and 9 November 2026). Two directions, each the regression vector of one
    PLS-DA model on the full 32³ latent grid, fitted once on all 12 open boxes: the cavity-free
    browning ≥ 2 model (5 LVs, out-of-fold AUC 0.848) and the general `binary_23` model (4 LVs,
    0.887). Height-radius rows cannot be decoded, so they give no direction. Both are applied to
    the same four fruit.

    **This version: section 1 only.** Which four fruit, and their CT, for Ruben to check before
    any decode (one decode took about 6 minutes on the Mac CPU).

    **Inputs.** The fruit table, `derived/volume_decisions.csv`, the CT volumes of the four
    fruit, the saved out-of-fold predictions of the two latent runs. The sealed test boxes are
    not loaded.

    **Outputs.** `derived/figures/17_counterfactual_fruit.png`.

    **Status.** draft, 9 November 2026.
    """)
    return


@app.cell(hide_code=True)
def _():
    cfg = load_config()
    return (cfg,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. The four fruit

    Rule (`pearlatent.counterfactual.draw_example_fruit`, seed 369): among the 315 open fruit,
    2 with browning 3 and cavity 0, then 2 with defective grade 0, every fruit from its own box.
    A fruit Ruben rejects on its CT is skipped and the next one of the same shuffled list takes
    its place (rejected: 2526_I09, browning not visible). These are examples only.
    Beside each fruit: its out-of-fold prediction in the two latent models whose direction will
    be used (average of the 20 repeats; a cavity-free prediction exists for cavity-free fruit only).
    """)
    return


@app.cell(hide_code=True)
def _(cfg):
    _open_2526 = open_fruit(cfg).filter(pl.col("season") == "2526")
    _kept = drop_faulty_volumes(
        _open_2526, cfg.path("derived_dir") / "volume_decisions.csv"
    )
    print(f"open 2526 fruit {_open_2526.height}, after faulty volumes {_kept.height}")
    # Rejected by Ruben on 9 November 2026 after seeing the CT: 2526_I09 (grade 3, but no
    # visible browning on the sections; the neck end also looked cut).
    REJECTED = ("2526_I09",)
    _drawn = draw_example_fruit(_kept, seed=DRAW_SEED, exclude=REJECTED)

    _results = cfg.path("results_dir")
    _general = pl.read_parquet(
        _results / PLSDA_BINARY23_RUN_DIRS["hugo_vae"] / "predictions.parquet"
    ).select("fruit_id", prediction_binary_23="prediction")
    _cavity_free_run = latest_run_dir(
        _results, "plsda_browning_2plus_cavity_free_hugo_vae"
    )
    print(f"cavity-free run: {_cavity_free_run.name}")
    _cavity_free = pl.read_parquet(_cavity_free_run / "predictions.parquet").select(
        "fruit_id", prediction_cavity_free="prediction"
    )
    examples = _drawn.join(_general, on="fruit_id", how="left", validate="1:1").join(
        _cavity_free, on="fruit_id", how="left", validate="1:1"
    )
    examples.select(
        "group",
        "fruit_id",
        "box",
        "browning",
        "cavity",
        "position",
        pl.col("prediction_binary_23").round(3),
        pl.col("prediction_cavity_free").round(3),
        "stalk_length",
        "body_voxels",
    )
    return REJECTED, examples


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. Their CT: is anything wrong with these fruit?

    Per fruit (one row): the two long sections through the centre of the body (they contain the
    stem-calyx axis; calyx end at the bottom) and the cross-section at the body centre. Min-max
    scaled as the encoder sees it, grey scale 0 to 1. Look for: a fruit that does not show its
    grade (browning 3 should be visible as darker tissue around the core), a cut or a stray
    object that the faulty-volume check missed.
    """)
    return


@app.cell(hide_code=True)
def _(cfg, examples):
    _cells = np.load(
        cfg.path("derived_dir") / ENCODING_RUN_DIR / "masks.npy", mmap_mode="r"
    )
    _titles = [
        "long section, first axis fixed",
        "long section, second axis fixed",
        "cross-section",
    ]
    fruit_figure, _axes = plt.subplots(
        examples.height, 3, figsize=(9, 3.1 * examples.height), squeeze=False
    )
    for _i, _row in enumerate(examples.iter_rows(named=True)):
        _path = cfg.path(f"ct_dir_{_row['season']}") / f"{_row['file_stem']}.nii"
        _scaled = scale_minmax(np.asarray(nib.load(_path).dataobj))
        _body = body_voxel_mask(_scaled, np.asarray(_cells[_row["position"]]))
        _planes = central_planes(_scaled, body_centre(_body))
        for _j in range(3):
            _ax = _axes[_i, _j]
            # Long sections: the stem-calyx axis (third array axis) runs upwards.
            _image = _planes[_j].T if _j < 2 else _planes[_j]
            _ax.imshow(_image, cmap="gray", vmin=0, vmax=1, origin="lower")
            _ax.set_xticks([])
            _ax.set_yticks([])
            if _i == 0:
                _ax.set_title(_titles[_j], fontsize=9)
        _axes[_i, 0].set_ylabel(
            f"{_row['fruit_id']}\n{_row['group']} (browning {_row['browning']},"
            f" cavity {_row['cavity']})",
            fontsize=9,
        )
    fruit_figure.tight_layout()
    _out = cfg.path("derived_dir") / "figures" / "17_counterfactual_fruit.png"
    fruit_figure.savefig(_out, dpi=120, bbox_inches="tight")
    print(f"saved {_out}")
    fruit_figure
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3. Steps along the model direction

    Made by `python -m pearlatent.counterfactual_run` (one decode about 5 minutes). t = 0 is the
    mean score of grade 0 fruit, t = 1 that of grade 3 (`pearlatent.counterfactual.GradeScale`).
    A step moves `mu` only along the regression vector b of the model, by the smallest change
    that gives the target score.

    Table: the target score of each step, and the score of the decoded image after encoding it
    again. "plain" is the reconstruction without a shift; its re-encoded score shows what the
    decode-encode round trip alone does. Figure, per step: the long section through the body
    centre, and the step minus the plain reconstruction (blue = brighter, red = darker than the
    plain reconstruction), on one colour scale for all steps, set by the voxels inside the fruit
    (CT above 0.1) so that a change outside the fruit does not hide the change inside it.
    """)
    return


@app.cell(hide_code=True)
def _(cfg):
    # Every counterfactual run folder (20 November 2026: one fruit can have several runs, for
    # example the EPO steps and an extra run at the decision boundary; all stay selectable).
    _runs = sorted(
        _p.name
        for _p in (cfg.path("derived_dir") / "counterfactuals").glob("*_*")
        if (_p / "run.json").exists()
    )
    counterfactual_choice = mo.ui.dropdown(
        options=_runs,
        value=_runs[0] if _runs else None,
        label="Run (timestamp, fruit, direction)",
    )
    counterfactual_choice
    return (counterfactual_choice,)


@app.cell(hide_code=True)
def _(cfg, counterfactual_choice):
    mo.stop(counterfactual_choice.value is None, mo.md("_No counterfactual run yet._"))
    _run = cfg.path("derived_dir") / "counterfactuals" / counterfactual_choice.value
    _record = json.loads((_run / "run.json").read_text())["params"]
    steps = pl.read_parquet(_run / "steps.parquet")
    _scale = (_record["grade0_mean_score"], _record["grade3_mean_score"])
    steps_shown = steps.with_columns(
        t_reencoded=(pl.col("score_reencoded") - _scale[0]) / (_scale[1] - _scale[0])
    )
    print(
        f"run {_run.name}; start score {_record['start_score']:.3f};"
        f" grade 0 mean {_scale[0]:.3f}, grade 3 mean {_scale[1]:.3f}"
    )
    counterfactual_run = _run
    counterfactual_record = _record
    steps_shown
    return counterfactual_record, counterfactual_run, steps


@app.cell(hide_code=True)
def _(cfg, counterfactual_record, counterfactual_run, steps):
    with np.load(
        cfg.root / counterfactual_record["plain_reconstruction"] / "volumes.npz"
    ) as _f:
        _k = list(_f["fruit_id"]).index(counterfactual_record["fruit"])
        _ct = _f["ct"][_k].astype(np.float32)
        _plain = _f["reconstruction"][_k].astype(np.float32)
    _images = {"CT": _ct, "plain reconstruction": _plain}
    for _t in steps.filter(pl.col("t_target").is_not_null())["t_target"].to_list():
        _file = counterfactual_run / f"step_t{_t}.npz"
        if _file.exists():
            with np.load(_file) as _f:
                _images[f"t = {_t}"] = _f["reconstruction"].astype(np.float32)
    # Long section through the body centre (first array axis fixed), stem-calyx axis upwards.
    _centre = body_centre(_ct > 0.1)
    _steps = [_k for _k in _images if _k.startswith("t =")]
    _inside = _ct[_centre[0]] > 0.1
    _limit = max(
        [
            float(np.abs(_images[_s] - _plain)[_centre[0]][_inside].max())
            for _s in _steps
        ]
        + [1e-3]
    )
    step_figure, _axes = plt.subplots(
        2, len(_images), figsize=(2.6 * len(_images), 5.6), squeeze=False
    )
    for _j, (_name, _volume) in enumerate(_images.items()):
        _axes[0, _j].imshow(
            _volume[_centre[0]].T, cmap="gray", vmin=0, vmax=1, origin="lower"
        )
        _axes[0, _j].set_title(_name, fontsize=9)
        if _name in _steps:
            _shown = _axes[1, _j].imshow(
                (_volume - _plain)[_centre[0]].T,
                cmap="RdBu",
                vmin=-_limit,
                vmax=_limit,
                origin="lower",
            )
        else:
            _axes[1, _j].axis("off")
    for _ax in _axes.ravel():
        _ax.set_xticks([])
        _ax.set_yticks([])
    if _steps:
        step_figure.colorbar(_shown, ax=_axes[1, :], shrink=0.8, label="step − plain")
    _out = cfg.path("derived_dir") / "figures" / f"17_{counterfactual_run.name}.png"
    step_figure.savefig(_out, dpi=120, bbox_inches="tight")
    print(f"saved {_out}; colour limit ±{_limit:.3f}")
    step_figure
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 4. All runs together: asked score against the score the decoded fruit gets

    Every run in `derived/counterfactuals`, in one table (20 November 2026: the old cavity-free
    direction, 5 LVs, and the EPO cavity-free direction, 4 LVs, each at t = 0.5 and at the far
    grade mean; plus the EPO flip, target score = 2 × 0.5 − start score, the mirror image of the
    fruit in the decision boundary).

    Per step:

    - **asked score**: the score the shifted latent `mu` has (the target; exact by construction);
    - **got score**: the score of the decoded image after encoding it again;
    - **asked change** = asked score − start score; **got change** = got score − the re-encoded
      score of the plain reconstruction. The second difference removes what the decode-encode
      round trip alone does to the score, so the two changes compare like with like;
    - **share kept** = got change / asked change: 1 means the decoded image carries the whole
      step, 0 means the decoder ignored it.
    """)
    return


@app.cell(hide_code=True)
def _(cfg, examples):
    # One row per step of every run. The plain row of each run gives the round-trip baseline.
    _rows = []
    for _run in sorted((cfg.path("derived_dir") / "counterfactuals").glob("*_*")):
        if not (_run / "run.json").exists():
            continue
        _record = json.loads((_run / "run.json").read_text())["params"]
        _steps = pl.read_parquet(_run / "steps.parquet")
        _plain = _steps.filter(pl.col("step") == "plain")["score_reencoded"].item()
        if not _record.get("epo"):
            _kind = f"no EPO ({_record['n_lv']} LVs)"
        elif _record.get("flip"):
            _kind = "EPO flip"
        else:
            _kind = f"EPO ({_record['n_lv']} LVs)"
        for _step in _steps.filter(pl.col("step") != "plain").iter_rows(named=True):
            _rows.append(
                {
                    "fruit_id": _record["fruit"],
                    "kind": _kind,
                    "run": _run.name,
                    "t_target": _step["t_target"],
                    "start_score": _record["start_score"],
                    "plain_reencoded": _plain,
                    "asked_score": _step["score_target"],
                    "got_score": _step["score_reencoded"],
                    "grade0_mean": _record["grade0_mean_score"],
                    "grade3_mean": _record["grade3_mean_score"],
                }
            )
    step_table = (
        pl.DataFrame(_rows)
        .join(
            examples.select("fruit_id", "group"),
            on="fruit_id",
            how="left",
            validate="m:1",
        )
        .with_columns(
            asked_change=pl.col("asked_score") - pl.col("start_score"),
            got_change=pl.col("got_score") - pl.col("plain_reencoded"),
            t_got=(pl.col("got_score") - pl.col("grade0_mean"))
            / (pl.col("grade3_mean") - pl.col("grade0_mean")),
        )
        .with_columns(share_kept=pl.col("got_change") / pl.col("asked_change"))
    )
    print(f"{step_table.height} steps from {step_table['run'].n_unique()} runs")
    step_table.select(
        "fruit_id",
        "group",
        "kind",
        "t_target",
        pl.col("t_got").round(2),
        pl.col("start_score").round(3),
        pl.col("plain_reencoded").round(3),
        pl.col("asked_score").round(3),
        pl.col("got_score").round(3),
        pl.col("share_kept").round(2),
    ).sort("fruit_id", "kind", "t_target")
    return (step_table,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The two figures below show the EPO direction only. The old direction (no EPO, 5 LVs) gives
    practically the same scores (within 0.02 in t) and the same images; its rows stay in the
    table above. Three steps per fruit:

    - **halfway**: to t = 0.5;
    - **to the other grade**: to the mean score of the other grade (t = 0 for a grade 3 fruit,
      t = 1 for a grade 0 fruit);
    - **mirrored**: to the mirror image of the fruit's score in the decision boundary
      (score 0.5).

    t is the score on a scale where 0 is the grade 0 mean score and 1 the grade 3 mean score.
    """)
    return


@app.cell(hide_code=True)
def _(step_table):
    # The EPO steps only, with a plain name per step (see the text above).
    _scale = step_table.select("grade0_mean", "grade3_mean").row(0)
    epo_steps = step_table.filter(pl.col("kind").str.starts_with("EPO")).with_columns(
        step_name=pl.when(pl.col("kind") == "EPO flip")
        .then(pl.lit("mirrored"))
        .when(pl.col("t_target") == 0.5)
        .then(pl.lit("halfway"))
        .otherwise(pl.lit("to the other grade")),
        t_start=(pl.col("start_score") - _scale[0]) / (_scale[1] - _scale[0]),
    )
    STEP_NAMES = ("halfway", "to the other grade", "mirrored")
    print(f"EPO steps: {epo_steps.height} of {step_table.height}")
    return STEP_NAMES, epo_steps


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Per fruit and step: black bar = the fruit's own score; grey line and open circle = the step
    asked for; blue dot = the score of the decoded image after encoding it again. Dashed line:
    the decision boundary. A blue dot on the open circle would mean the decoded fruit got what
    was asked.
    """)
    return


@app.cell(hide_code=True)
def _(STEP_NAMES, cfg, epo_steps, examples):
    _fruit_order = examples["fruit_id"].to_list()
    _groups = dict(examples.select("fruit_id", "group").iter_rows())
    _scale = epo_steps.select("grade0_mean", "grade3_mean").row(0)
    _boundary_t = (0.5 - _scale[0]) / (
        _scale[1] - _scale[0]
    )  # score 0.5 on the t scale
    _asked_colour, _got_colour = "#8a8a8a", "#2a78d6"

    score_figure, _axes = plt.subplots(
        2, 2, figsize=(10, 4.6), sharex=True, sharey=True
    )
    for _ax, _fruit in zip(_axes.ravel(), _fruit_order):
        for _y, _name in enumerate(STEP_NAMES):
            _match = epo_steps.filter(
                (pl.col("fruit_id") == _fruit) & (pl.col("step_name") == _name)
            )
            if _match.height != 1:
                continue
            _step = _match.row(0, named=True)
            _ax.plot(
                [_step["t_start"], _step["t_target"]],
                [_y, _y],
                color=_asked_colour,
                lw=2,
            )
            _ax.plot(
                _step["t_start"], _y, marker="|", color="black", markersize=14, mew=2
            )
            _ax.scatter(
                _step["t_target"],
                _y,
                s=70,
                facecolor="white",
                edgecolor=_asked_colour,
                linewidth=2,
                zorder=3,
            )
            _ax.scatter(_step["t_got"], _y, s=70, color=_got_colour, zorder=4)
        _ax.axvline(_boundary_t, color="0.3", lw=1, ls="--", zorder=0)
        _ax.set_yticks(range(len(STEP_NAMES)), STEP_NAMES, fontsize=9)
        _ax.set_ylim(len(STEP_NAMES) - 0.5, -0.5)
        _ax.set_title(f"{_fruit} ({_groups[_fruit]})", fontsize=10, loc="left")
        _ax.grid(axis="x", color="0.92")
        for _side in ("top", "right"):
            _ax.spines[_side].set_visible(False)
    for _ax in _axes[-1]:
        _ax.set_xlabel("t")
    _handles = [
        plt.Line2D(
            [],
            [],
            color="black",
            marker="|",
            ls="",
            markersize=12,
            mew=2,
            label="fruit",
        ),
        plt.Line2D(
            [],
            [],
            color=_asked_colour,
            marker="o",
            markerfacecolor="white",
            markeredgewidth=2,
            lw=2,
            label="asked",
        ),
        plt.Line2D([], [], color=_got_colour, marker="o", ls="", label="got"),
        plt.Line2D([], [], color="0.3", ls="--", label="decision boundary"),
    ]
    score_figure.legend(
        handles=_handles, loc="upper center", ncol=4, fontsize=9, frameon=False
    )
    score_figure.tight_layout(rect=(0, 0, 1, 0.93))
    _out = cfg.path("derived_dir") / "figures" / "17_asked_against_got.png"
    score_figure.savefig(_out, dpi=130, bbox_inches="tight")
    print(f"saved {_out}")
    score_figure
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 5. The images

    One row per fruit, EPO direction. Columns: the CT, the plain reconstruction, then each step
    minus the plain reconstruction (long section through the body centre, calyx end at the
    bottom; blue = brighter, red = darker than the plain reconstruction). One colour scale per
    row, set by the voxels inside the fruit. Under each map: the t asked for and the t got.
    """)
    return


@app.cell(hide_code=True)
def _(STEP_NAMES, cfg, epo_steps, examples):
    _fruit_order = examples["fruit_id"].to_list()
    _groups = dict(examples.select("fruit_id", "group").iter_rows())
    _n_columns = 2 + len(STEP_NAMES)
    image_figure, _axes = plt.subplots(
        len(_fruit_order),
        _n_columns,
        figsize=(2.2 * _n_columns, 2.7 * len(_fruit_order)),
        squeeze=False,
    )
    for _i, _fruit in enumerate(_fruit_order):
        _fruit_steps = epo_steps.filter(pl.col("fruit_id") == _fruit)
        _record = json.loads(
            (
                cfg.path("derived_dir")
                / "counterfactuals"
                / _fruit_steps["run"][0]
                / "run.json"
            ).read_text()
        )["params"]
        with np.load(cfg.root / _record["plain_reconstruction"] / "volumes.npz") as _f:
            _k = list(_f["fruit_id"]).index(_fruit)
            _ct = _f["ct"][_k].astype(np.float32)
            _plain = _f["reconstruction"][_k].astype(np.float32)
        _centre = body_centre(_ct > 0.1)
        _inside = _ct[_centre[0]] > 0.1
        _maps = {}
        for _step in _fruit_steps.iter_rows(named=True):
            _file = (
                cfg.path("derived_dir")
                / "counterfactuals"
                / _step["run"]
                / f"step_t{_step['t_target']}.npz"
            )
            with np.load(_file) as _f:
                _section = _f["reconstruction"].astype(np.float32)[_centre[0]]
            _maps[_step["step_name"]] = (
                _section - _plain[_centre[0]],
                _step["t_target"],
                _step["t_got"],
            )
        _limit = max(
            [float(np.abs(_d[_inside]).max()) for _d, _, _ in _maps.values()] + [1e-3]
        )
        for _j, (_name, _image) in enumerate(
            [("CT", _ct), ("plain reconstruction", _plain)]
        ):
            _axes[_i, _j].imshow(
                _image[_centre[0]].T, cmap="gray", vmin=0, vmax=1, origin="lower"
            )
            if _i == 0:
                _axes[_i, _j].set_title(_name, fontsize=9)
        for _j, _name in enumerate(STEP_NAMES, start=2):
            _ax = _axes[_i, _j]
            if _i == 0:
                _ax.set_title(_name, fontsize=9)
            if _name not in _maps:
                _ax.axis("off")
                continue
            _difference, _t_asked, _t_got = _maps[_name]
            _shown = _ax.imshow(
                _difference.T, cmap="RdBu", vmin=-_limit, vmax=_limit, origin="lower"
            )
            _ax.set_xlabel(f"asked {_t_asked:.2f}, got {_t_got:.2f}", fontsize=8)
        # The bar takes its width from the whole row, so the last column keeps its size.
        image_figure.colorbar(
            _shown, ax=_axes[_i, :].tolist(), shrink=0.8, pad=0.01, label="step − plain"
        )
        _axes[_i, 0].set_ylabel(f"{_fruit}\n{_groups[_fruit]}", fontsize=9)
    for _ax in _axes.ravel():
        _ax.set_xticks([])
        _ax.set_yticks([])
    _out = cfg.path("derived_dir") / "figures" / "17_all_runs_images.png"
    image_figure.savefig(_out, dpi=130, bbox_inches="tight")
    print(f"saved {_out}")
    image_figure
    return


if __name__ == "__main__":
    app.run()

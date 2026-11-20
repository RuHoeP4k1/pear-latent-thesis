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
    _cells = np.load(cfg.path("derived_dir") / ENCODING_RUN_DIR / "masks.npy", mmap_mode="r")
    _titles = ["long section, first axis fixed", "long section, second axis fixed", "cross-section"]
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
    mo.stop(
        counterfactual_choice.value is None, mo.md("_No counterfactual run yet._")
    )
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
    with np.load(cfg.root / counterfactual_record["plain_reconstruction"] / "volumes.npz") as _f:
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
        [float(np.abs(_images[_s] - _plain)[_centre[0]][_inside].max()) for _s in _steps]
        + [1e-3]
    )
    step_figure, _axes = plt.subplots(
        2, len(_images), figsize=(2.6 * len(_images), 5.6), squeeze=False
    )
    for _j, (_name, _volume) in enumerate(_images.items()):
        _axes[0, _j].imshow(_volume[_centre[0]].T, cmap="gray", vmin=0, vmax=1, origin="lower")
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
        .join(examples.select("fruit_id", "group"), on="fruit_id", how="left", validate="m:1")
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
    Left: each line starts at the fruit's plain reconstruction (x = its start score, y = the
    re-encoded score of the plain reconstruction) and runs to each step (x = asked, y = got). On
    the grey diagonal the decoded fruit gets exactly what was asked. The dashed lines are the
    decision boundary (score 0.5): a step that changes the call ends in the other quadrant
    (top right = non-consumable, bottom left = consumable). Right: asked change against got
    change; the slope of a point from the origin is the share kept.
    """)
    return


@app.cell(hide_code=True)
def _(cfg, examples, step_table):
    # Colour = fruit (fixed order of the example table), marker = kind of run.
    _fruit_order = examples["fruit_id"].to_list()
    _colours = dict(zip(_fruit_order, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]))
    _markers = {"no EPO (5 LVs)": "o", "EPO (4 LVs)": "s", "EPO flip": "*"}
    _boundary = 0.5  # pearlatent.counterfactual_run CUTOFF
    score_figure, (_ax_path, _ax_change) = plt.subplots(1, 2, figsize=(12, 5.4))

    _limits = (-0.35, 1.3)
    _ax_path.plot(_limits, _limits, color="0.75", lw=1, zorder=0)
    _ax_path.axhline(_boundary, color="0.4", lw=1, ls="--", zorder=0)
    _ax_path.axvline(_boundary, color="0.4", lw=1, ls="--", zorder=0)
    for (_fruit, _kind), _group in step_table.group_by(["fruit_id", "kind"]):
        _start = (_group["start_score"][0], _group["plain_reencoded"][0])
        for _step in _group.iter_rows(named=True):
            _ax_path.plot(
                [_start[0], _step["asked_score"]],
                [_start[1], _step["got_score"]],
                color=_colours[_fruit],
                lw=1.5,
                alpha=0.6,
                ls=":" if _kind == "EPO flip" else "-",
            )
            _ax_path.scatter(
                _step["asked_score"],
                _step["got_score"],
                color=_colours[_fruit],
                marker=_markers[_kind],
                s=110 if _kind == "EPO flip" else 55,
                edgecolor="white",
                linewidth=1,
                zorder=3,
            )
        _ax_path.scatter(*_start, color=_colours[_fruit], marker="x", s=50, zorder=3)
    _ax_path.set_xlim(_limits)
    _ax_path.set_ylim(_limits)
    _ax_path.set_aspect("equal")
    _ax_path.set_xlabel("asked score (shifted latent)")
    _ax_path.set_ylabel("got score (decoded image, encoded again)")
    _ax_path.set_title("Where the decoded fruit lands", fontsize=10)

    _span = 1.2
    _ax_change.plot((-_span, _span), (-_span, _span), color="0.75", lw=1, zorder=0)
    _ax_change.axhline(0, color="0.85", lw=1, zorder=0)
    _ax_change.axvline(0, color="0.85", lw=1, zorder=0)
    for _step in step_table.iter_rows(named=True):
        _ax_change.scatter(
            _step["asked_change"],
            _step["got_change"],
            color=_colours[_step["fruit_id"]],
            marker=_markers[_step["kind"]],
            s=110 if _step["kind"] == "EPO flip" else 55,
            edgecolor="white",
            linewidth=1,
            zorder=3,
        )
    _ax_change.set_xlim(-_span, _span)
    _ax_change.set_ylim(-_span, _span)
    _ax_change.set_aspect("equal")
    _ax_change.set_xlabel("asked change (asked − start score)")
    _ax_change.set_ylabel("got change (got − plain re-encoded score)")
    _ax_change.set_title("Share of the step that survives decoding", fontsize=10)

    # Legend: fruit by colour, kind by marker.
    _groups = dict(examples.select("fruit_id", "group").iter_rows())
    _handles = [
        plt.Line2D([], [], color=_colours[_f], lw=2, label=f"{_f} ({_groups[_f]})")
        for _f in _fruit_order
    ] + [
        plt.Line2D([], [], color="0.3", marker=_m, ls="", markersize=8, label=_k)
        for _k, _m in _markers.items()
    ] + [plt.Line2D([], [], color="0.3", marker="x", ls="", label="plain (start)")]
    score_figure.legend(handles=_handles, loc="lower center", ncol=4, fontsize=8, frameon=False)
    score_figure.tight_layout(rect=(0, 0.1, 1, 1))
    _out = cfg.path("derived_dir") / "figures" / "17_asked_against_got.png"
    score_figure.savefig(_out, dpi=130, bbox_inches="tight")
    print(f"saved {_out}")
    score_figure
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 5. The images of all runs

    One row per fruit. Columns: the CT, the plain reconstruction, then every step minus the
    plain reconstruction (long section through the body centre, calyx end at the bottom;
    blue = brighter, red = darker than the plain reconstruction). One colour scale per row, set
    by the voxels inside the fruit, so the old and the EPO direction of one fruit compare
    directly. Above each map: the asked t and the t the decoded image gets.
    """)
    return


@app.cell(hide_code=True)
def _(cfg, examples, step_table):
    _columns = [
        ("no EPO (5 LVs)", "half"),
        ("no EPO (5 LVs)", "far"),
        ("EPO (4 LVs)", "half"),
        ("EPO (4 LVs)", "far"),
        ("EPO flip", "flip"),
    ]
    _fruit_order = examples["fruit_id"].to_list()
    _groups = dict(examples.select("fruit_id", "group").iter_rows())
    image_figure, _axes = plt.subplots(
        len(_fruit_order),
        2 + len(_columns),
        figsize=(2.1 * (2 + len(_columns)), 2.6 * len(_fruit_order)),
        squeeze=False,
    )
    for _i, _fruit in enumerate(_fruit_order):
        _fruit_steps = step_table.filter(pl.col("fruit_id") == _fruit)
        _record = json.loads(
            (
                cfg.path("derived_dir") / "counterfactuals" / _fruit_steps["run"][0] / "run.json"
            ).read_text()
        )["params"]
        with np.load(cfg.root / _record["plain_reconstruction"] / "volumes.npz") as _f:
            _k = list(_f["fruit_id"]).index(_fruit)
            _ct = _f["ct"][_k].astype(np.float32)
            _plain = _f["reconstruction"][_k].astype(np.float32)
        _centre = body_centre(_ct > 0.1)
        _inside = _ct[_centre[0]] > 0.1
        # Which step goes in which column: t = 0.5 is "half", the other non-flip step "far".
        _maps = {}
        for _step in _fruit_steps.iter_rows(named=True):
            _slot = (
                "flip"
                if _step["kind"] == "EPO flip"
                else ("half" if _step["t_target"] == 0.5 else "far")
            )
            _file = (
                cfg.path("derived_dir")
                / "counterfactuals"
                / _step["run"]
                / f"step_t{_step['t_target']}.npz"
            )
            with np.load(_file) as _f:
                _difference = _f["reconstruction"].astype(np.float32)[_centre[0]] - _plain[
                    _centre[0]
                ]
            _maps[(_step["kind"], _slot)] = (_difference, _step["t_target"], _step["t_got"])
        _limit = max(
            [float(np.abs(_d[_inside]).max()) for _d, _, _ in _maps.values()] + [1e-3]
        )
        for _j, (_name, _image) in enumerate([("CT", _ct), ("plain", _plain)]):
            _axes[_i, _j].imshow(
                _image[_centre[0]].T, cmap="gray", vmin=0, vmax=1, origin="lower"
            )
            if _i == 0:
                _axes[_i, _j].set_title(_name, fontsize=9)
        for _j, _key in enumerate(_columns, start=2):
            _ax = _axes[_i, _j]
            if _i == 0:
                _ax.set_title(f"{_key[0]}\n{_key[1]}", fontsize=9)
            if _key not in _maps:
                _ax.axis("off")
                continue
            _difference, _t_asked, _t_got = _maps[_key]
            _shown = _ax.imshow(
                _difference.T, cmap="RdBu", vmin=-_limit, vmax=_limit, origin="lower"
            )
            _ax.text(
                0.5,
                0.02,
                f"t asked {_t_asked:.2f}\nt got {_t_got:.2f}",
                transform=_ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=8,
            )
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

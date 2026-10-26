import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="07 Latent exploration")

with app.setup:
    import altair as alt
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    from sklearn.decomposition import PCA

    from pearlatent.config import load_config
    from pearlatent.exclusions import read_volume_decisions
    from pearlatent.labels import load_labels


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 07 Latent exploration: what organises the latent space?

    **Question.** Which known properties of the fruit (orchard box, season, grade, cause, stalk,
    size, intensity) go with the main structure of the latent, and how strongly?

    **Plan.** `docs/exploration_plan.md` (23 October 2026). t-SNE, UMAP, the MDS map of group
    centres and PERMANOVA with PERMDISP were removed on 26 October 2026 (Ruben): they repeated
    the stalk finding of section 1 and are not used in the thesis.

    **Inputs.** Encoding run `20261013T171306Z_hugo_vae` (`latents.npy`, `manifest.parquet`,
    `reconstruction.parquet`), both label files, `derived/descriptors/20261023T155219Z_descriptors/`,
    `derived/volume_decisions.csv` when it exists.

    **Fruit.** The 346 open 2526 fruit; Hugo's 66 2425 validation fruit for the season
    comparison. Not used: his 528 training fruit (charter rule 2) and all sealed fruit.

    **Outputs.** Figures and tables only.

    **Status.** draft
    """)
    return


@app.cell(hide_code=True)
def _():
    cfg = load_config()
    run_dir = (
        cfg.path("derived_dir") / "latents" / "hugo_vae" / "20261013T171306Z_hugo_vae"
    )
    descriptor_dir = (
        cfg.path("derived_dir") / "descriptors" / "20261023T155219Z_descriptors"
    )
    return cfg, descriptor_dir, run_dir


@app.cell(hide_code=True)
def _(cfg, descriptor_dir, run_dir):
    _manifest = pl.read_parquet(run_dir / "manifest.parquet").with_row_index("position")
    _labels = load_labels(cfg).select("fruit_id", "browning", "cavity")
    _descriptors = pl.read_parquet(descriptor_dir / "descriptors.parquet")
    _reconstruction = pl.read_parquet(run_dir / "reconstruction.parquet").select(
        "fruit_id", "mae_core"
    )
    _all = (
        _manifest.drop("storage")
        .join(
            load_labels(cfg).select("fruit_id", "storage"),
            on="fruit_id",
            validate="1:1",
        )
        .join(_labels, on="fruit_id", validate="1:1")
        .join(_descriptors, on="fruit_id", validate="1:1")
        .join(_reconstruction, on="fruit_id", how="left", validate="1:1")
        .with_columns(defective=pl.max_horizontal("browning", "cavity"))
        .with_columns(
            cause=pl.when(pl.col("defective") == 0)
            .then(pl.lit("none"))
            .when(pl.col("cavity") == 0)
            .then(pl.lit("browning only"))
            .when(pl.col("browning") == 0)
            .then(pl.lit("cavity only"))
            .otherwise(pl.lit("both"))
        )
    )
    # 2526 open fruit and Hugo's 2425 validation fruit (section "Fruit" above).
    _kept = _all.filter(
        (pl.col("season") == "2526") | (pl.col("hugo_split") == "validation")
    )
    print(f"open fruit: {_all.height}; 2526 plus 2425 validation: {_kept.height}")

    _decisions_file = cfg.path("derived_dir") / "volume_decisions.csv"
    if _decisions_file.exists():
        _faulty = read_volume_decisions(_decisions_file).filter(
            pl.col("decision") != "keep"
        )
        fruit = _kept.join(_faulty.select("fruit_id"), on="fruit_id", how="anti")
        print(f"faulty volumes dropped: {_kept.height - fruit.height}")
    else:
        fruit = _kept
        print("no decisions file yet: no faulty volume dropped")
    print(f"fruit in this notebook: {fruit.height}")
    fruit.group_by("season").len().sort("season")
    return (fruit,)


@app.cell(hide_code=True)
def _(fruit, run_dir):
    # Rows of latents.npy in the order of `fruit`, flattened to 32,768 numbers per fruit.
    _latents = np.load(run_dir / "latents.npy", mmap_mode="r")
    flat = np.asarray(_latents[fruit["position"].to_numpy()]).reshape(fruit.height, -1)
    is_2526 = (fruit["season"] == "2526").to_numpy()

    # PCA as in notebook 05: mean-centred, not scaled, exact solver, fitted on the 2526 fruit;
    # every fruit projected.
    _pca = PCA(n_components=50, svd_solver="full").fit(flat[is_2526])
    pc_scores = _pca.transform(flat)
    print(
        f"latents {flat.shape}; PCA fitted on {is_2526.sum()} fruit; scores {pc_scores.shape}"
    )
    return is_2526, pc_scores


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. What do the first principal components follow?

    Spearman rank correlation of PC1 to PC10 (2526 fruit) with the fruit descriptors and the
    grades. A component that correlates strongly with stalk length or body size describes the
    shape of the fruit in the grid, not its inside.
    """)
    return


@app.cell(hide_code=True)
def _(fruit, is_2526, pc_scores):
    _properties = [
        "stalk_length",
        "body_voxels",
        "body_length",
        "diameter_to_length",
        "tissue_intensity_median",
        "cavity_fraction",
        "flat_face_share",
        "mae_core",
        "browning",
        "cavity",
        "defective",
    ]
    _in_2526 = fruit.filter(pl.Series(is_2526)).with_columns(
        pl.Series(f"PC{i + 1}", pc_scores[is_2526, i]) for i in range(10)
    )
    pc_correlations = pl.DataFrame(
        {
            "property": _properties,
            **{
                f"PC{i + 1}": [
                    round(
                        _in_2526.select(
                            pl.corr(f"PC{i + 1}", p, method="spearman")
                        ).item(),
                        2,
                    )
                    for p in _properties
                ]
                for i in range(10)
            },
        }
    )
    pc_correlations
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. Score plot PC1 against PC2, four colourings side by side

    The same PCA as in section 1 (fitted on the 2526 fruit after the exclusions). Panels 1 to 3:
    2526 fruit only, coloured by stalk length, body voxels (both in grid voxels, so relative to
    the 128-voxel length of each fruit) and defective grade. Panel 4: Hugo's 2425 validation
    fruit projected onto the same components, coloured by season. For the meeting with Hugo
    (`docs/meeting_2026-10-29.md`, Part 2 item 2): if the colour runs along PC1 in panels 1
    and 2 but not in panel 3, the first component describes the fruit shape in the grid,
    not the defect.
    """)
    return


@app.cell(hide_code=True)
def _(fruit, is_2526, pc_scores):
    _x, _y = pc_scores[:, 0], pc_scores[:, 1]
    _fig, _axes = plt.subplots(1, 4, figsize=(20, 4.8), constrained_layout=True)
    _continuous = [
        ("stalk_length", "stalk length (voxels)"),
        ("body_voxels", "body voxels"),
    ]
    for _ax, (_column, _label) in zip(_axes[:2], _continuous):
        _points = _ax.scatter(
            _x[is_2526],
            _y[is_2526],
            c=fruit.filter(pl.Series(is_2526))[_column].to_numpy(),
            cmap="viridis",
            s=12,
        )
        _fig.colorbar(_points, ax=_ax, label=_label)
        _ax.set_title(f"2526, coloured by {_label}")
    # Defective grade: one colour per grade, light to dark.
    _grade = fruit.filter(pl.Series(is_2526))["defective"].to_numpy()
    for _g, _colour in zip(range(4), ["#c6dbef", "#6baed6", "#2171b5", "#08306b"]):
        _axes[2].scatter(
            _x[is_2526][_grade == _g],
            _y[is_2526][_grade == _g],
            color=_colour,
            s=12,
            label=f"grade {_g}",
        )
    _axes[2].legend(fontsize=8)
    _axes[2].set_title("2526, coloured by defective grade")
    for _season, _colour in [("2526", "lightgrey"), ("2425", "darkorange")]:
        _in = (fruit["season"] == _season).to_numpy()
        _axes[3].scatter(_x[_in], _y[_in], color=_colour, s=12, label=_season)
    _axes[3].legend(fontsize=8)
    _axes[3].set_title("season (2425: Hugo's validation fruit, projected)")
    for _ax in _axes:
        _ax.set_xlabel("PC1")
        _ax.set_ylabel("PC2")
    _fig.suptitle(
        f"PCA of the latent, fitted on {int(is_2526.sum())} 2526 fruit after the exclusions"
    )
    pc_panel_figure = _fig
    pc_panel_figure
    return (pc_panel_figure,)


@app.cell(hide_code=True)
def _():
    save_pc_panel = mo.ui.run_button(label="Save the score plots to derived/figures/")
    save_pc_panel
    return (save_pc_panel,)


@app.cell(hide_code=True)
def _(cfg, pc_panel_figure, save_pc_panel):
    # Saved under derived/ (outside git).
    mo.stop(mo.running_in_notebook() and not save_pc_panel.value)
    _out = cfg.path("derived_dir") / "figures" / "07_pc1_pc2_panels.png"
    pc_panel_figure.savefig(_out, dpi=120, bbox_inches="tight")
    print(f"saved {_out}")
    return


if __name__ == "__main__":
    app.run()

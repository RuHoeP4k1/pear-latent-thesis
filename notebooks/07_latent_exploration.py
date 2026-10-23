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
    from sklearn.manifold import MDS, TSNE
    from statsmodels.stats.multitest import multipletests
    from umap import UMAP

    from pearlatent.config import load_config
    from pearlatent.exclusions import read_volume_decisions
    from pearlatent.labels import load_labels
    from pearlatent.permanova import permanova, permdisp, squared_distances


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 07 Latent exploration: what organises the latent space?

    **Question.** Which known properties of the fruit (orchard box, season, grade, cause, stalk,
    size, intensity) go with the main structure of the latent, and how strongly?

    **Plan and assumptions.** Written before this notebook was run: `docs/exploration_plan.md`
    (23 October 2026). Pictures (t-SNE, UMAP, MDS of group centres) are qualitative only
    (charter, "Do not"); the numbers come from PERMANOVA with PERMDISP.

    **Inputs.** Encoding run `20261013T171306Z_hugo_vae` (`latents.npy`, `manifest.parquet`,
    `reconstruction.parquet`), both label files, `derived/descriptors/20261023T155219Z_descriptors/`,
    `derived/volume_decisions.csv` when it exists.

    **Fruit.** The 346 open 2526 fruit; Hugo's 66 2425 validation fruit for the season
    comparison. Not used: his 528 training fruit (charter rule 2) and all sealed fruit.

    **Outputs.** Figures and tables only.

    **Status.** draft
    """)
    return


@app.cell
def _():
    cfg = load_config()
    run_dir = (
        cfg.path("derived_dir") / "latents" / "hugo_vae" / "20261013T171306Z_hugo_vae"
    )
    descriptor_dir = (
        cfg.path("derived_dir") / "descriptors" / "20261023T155219Z_descriptors"
    )
    # One seed for t-SNE, UMAP, MDS and the permutations (the project seed).
    SEED = 369
    return SEED, cfg, descriptor_dir, run_dir


@app.cell
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


@app.cell
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
    return flat, is_2526, pc_scores


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. What do the first principal components follow?

    Spearman rank correlation of PC1 to PC10 (2526 fruit) with the fruit descriptors and the
    grades. A component that correlates strongly with stalk length or body size describes the
    shape of the fruit in the grid, not its inside.
    """)
    return


@app.cell
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
    return (pc_correlations,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. t-SNE and UMAP (qualitative)

    Both on the 50 PCA scores of all fruit in this notebook. Settings fixed in
    `docs/exploration_plan.md`. Read only which fruit lie near which; distances between
    clusters and cluster sizes mean nothing in these maps.
    """)
    return


@app.cell
def _(SEED, fruit, pc_scores):
    _tsne = TSNE(
        n_components=2, perplexity=30, init="pca", random_state=SEED
    ).fit_transform(pc_scores)
    _umap = UMAP(
        n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=SEED
    ).fit_transform(pc_scores)
    embedded = fruit.with_columns(
        tsne_1=_tsne[:, 0],
        tsne_2=_tsne[:, 1],
        umap_1=_umap[:, 0],
        umap_2=_umap[:, 1],
    )
    return (embedded,)


@app.cell
def _():
    _categorical = [
        "season",
        "box",
        "storage",
        "defective",
        "cause",
        "browning",
        "cavity",
        "stalk_present",
    ]
    _continuous = [
        "stalk_length",
        "body_voxels",
        "diameter_to_length",
        "tissue_intensity_median",
        "flat_face_share",
        "mae_core",
    ]
    embedding_colour = mo.ui.dropdown(
        options=_categorical + _continuous, value="season", label="Colour by"
    )
    embedding_only_2526 = mo.ui.checkbox(value=False, label="2526 fruit only")
    continuous_colours = set(_continuous)
    mo.hstack([embedding_colour, embedding_only_2526], justify="start")
    return continuous_colours, embedding_colour, embedding_only_2526


@app.cell
def _(continuous_colours, embedded, embedding_colour, embedding_only_2526):
    _shown = (
        embedded.filter(pl.col("season") == "2526")
        if embedding_only_2526.value
        else embedded
    )
    _field = embedding_colour.value
    _type = "Q" if _field in continuous_colours else "N"
    _colour = alt.Color(
        f"{_field}:{_type}",
        scale=alt.Scale(scheme="viridis") if _type == "Q" else alt.Undefined,
    )
    _tooltip = ["fruit_id", "box", "storage", "browning", "cavity", "stalk_length"]

    def embedding_chart(prefix: str, title: str) -> alt.Chart:
        """Scatter plot of one embedding, coloured by the chosen field."""
        return (
            alt.Chart(_shown)
            .mark_circle(size=35, opacity=0.8)
            .encode(
                x=alt.X(f"{prefix}_1:Q", title=None, axis=None),
                y=alt.Y(f"{prefix}_2:Q", title=None, axis=None),
                color=_colour,
                tooltip=_tooltip,
            )
            .properties(width=380, height=340, title=title)
        )

    alt.hconcat(embedding_chart("tsne", "t-SNE"), embedding_chart("umap", "UMAP"))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3. PERMANOVA with PERMDISP

    Euclidean distances between the full latent grids. 9,999 permutations, seed 369; free for
    box and season, within box for grade, cause and stalk class (`docs/exploration_plan.md`,
    section 3). R² is the share of the sum of squared distances that the grouping explains.
    Holm's correction over the four planned 2526 tests. PERMDISP beside every test: if it is
    significant, part of the PERMANOVA result may be a difference in spread, not in position.
    """)
    return


@app.cell
def _(SEED, fruit, flat, is_2526):
    N_PERMUTATIONS = 9999
    _in_2526 = fruit.filter(pl.Series(is_2526))
    _x_2526 = flat[is_2526]
    _d2_2526 = squared_distances(_x_2526)

    # Stalk length in three classes at the tertiles of the 2526 fruit.
    _cuts = (
        _in_2526["stalk_length"].quantile(1 / 3),
        _in_2526["stalk_length"].quantile(2 / 3),
    )
    _stalk_class = np.digitize(_in_2526["stalk_length"].to_numpy(), _cuts, right=True)
    print(f"stalk tertile cuts (voxels): {_cuts}")

    _box = _in_2526["box"].to_numpy()
    _tests = [
        ("box", _box, None),
        ("defective grade", _in_2526["defective"].to_numpy(), _box),
        ("cause", _in_2526["cause"].to_numpy(), _box),
        ("stalk class (tertiles)", _stalk_class, _box),
    ]
    _rows = []
    for _name, _labels, _strata in _tests:
        _p = permanova(_d2_2526, _labels, _strata, N_PERMUTATIONS, SEED)
        _d = permdisp(_x_2526, _labels, _strata, N_PERMUTATIONS, SEED)
        _rows.append(
            {
                "fruit": "2526",
                "grouping": _name,
                "permutation": "free" if _strata is None else "within box",
                "groups": _p["groups"],
                "n": _p["n"],
                "pseudo_F": round(_p["pseudo_F"], 2),
                "R2": round(_p["R2"], 4),
                "p": _p["p"],
                "PERMDISP_F": round(_d["F"], 2),
                "PERMDISP_p": _d["p"],
            }
        )
    _holm = multipletests([r["p"] for r in _rows], method="holm")[1]
    for _row, _corrected in zip(_rows, _holm):
        _row["p_Holm"] = float(_corrected)

    # Season: descriptive only (confounded with scanner session, storage and stalk).
    _season = fruit["season"].to_numpy()
    _p = permanova(squared_distances(flat), _season, None, N_PERMUTATIONS, SEED)
    _d = permdisp(flat, _season, None, N_PERMUTATIONS, SEED)
    _rows.append(
        {
            "fruit": "2526 and 2425 validation",
            "grouping": "season (descriptive)",
            "permutation": "free",
            "groups": _p["groups"],
            "n": _p["n"],
            "pseudo_F": round(_p["pseudo_F"], 2),
            "R2": round(_p["R2"], 4),
            "p": _p["p"],
            "PERMDISP_F": round(_d["F"], 2),
            "PERMDISP_p": _d["p"],
            "p_Holm": None,
        }
    )
    permanova_table = pl.DataFrame(_rows)
    permanova_table
    return (permanova_table,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 4. Group centres in two dimensions (MDS)

    The mean latent of every box and defective grade with at least 3 fruit (2526), and the
    Euclidean distances between these means, drawn with metric multidimensional scaling.
    Colour: box; label: grade. If boxes dominate, points of one colour lie together; if grades
    dominate, points with the same number lie together.
    """)
    return


@app.cell
def _(SEED, fruit, flat, is_2526):
    MIN_FRUIT_PER_CENTRE = 3
    _in_2526 = fruit.filter(pl.Series(is_2526)).with_row_index("row_2526")
    _x = flat[is_2526]
    _cells = (
        _in_2526.group_by("box", "defective")
        .agg(pl.col("row_2526"), n=pl.len())
        .filter(pl.col("n") >= MIN_FRUIT_PER_CENTRE)
        .sort("box", "defective")
    )
    _centres = np.stack(
        [_x[rows].mean(axis=0) for rows in _cells["row_2526"].to_list()]
    )
    _distances = np.sqrt(squared_distances(_centres))
    _xy = MDS(
        n_components=2,
        dissimilarity="precomputed",
        metric=True,
        random_state=SEED,
        n_init=4,
    ).fit_transform(_distances)
    centre_map = _cells.drop("row_2526").with_columns(mds_1=_xy[:, 0], mds_2=_xy[:, 1])
    print(f"{centre_map.height} centres of box and grade with at least 3 fruit")

    _base = alt.Chart(centre_map).encode(
        x=alt.X("mds_1:Q", title="MDS 1"),
        y=alt.Y("mds_2:Q", title="MDS 2"),
    )
    _points = _base.mark_circle(size=120).encode(
        color=alt.Color("box:N"), tooltip=["box", "defective", "n"]
    )
    _text = _base.mark_text(dy=-11, fontSize=11).encode(text="defective:N")
    alt.layer(_points, _text).properties(width=520, height=420)
    return (centre_map,)


if __name__ == "__main__":
    app.run()

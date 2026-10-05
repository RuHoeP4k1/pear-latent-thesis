import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="01 Label inventory")

with app.setup:
    import marimo as mo
    import polars as pl

    from pearlatent.config import load_config
    from pearlatent.labels import (
        check_derived_labels,
        derive_labels,
        load_labels,
        read_label_file,
    )
    from pearlatent.volumes import list_volumes


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 01 Label inventory

    **Question.** What do the two label files contain, do they agree with the grading rubric and
    with their own derived columns, and which labelled fruit have a computed tomography (CT)
    volume?

    **Inputs.** `labels_2425`, `labels_2526` (label CSV files), `ct_dir_2425`, `ct_dir_2526`
    (volume folders; only the file names are read), all from `config/local.toml`.

    **Outputs.** None written yet. Tables shown in the notebook: fruit per box and browning grade
    per season, disagreements between derived and stored label columns, labelled fruit without a
    volume (17 expected in 2526) and volumes without a label.

    **Status.** draft
    """)
    return


@app.cell
def _():
    cfg = load_config()
    return (cfg,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. The two label files

    `read_label_file` stops if the header differs from the expected columns, if a fruit occurs
    twice, or if a grade lies outside the rubric range (browning and cavity 0 to 3, rot 0 to 1).
    Each fruit gets `fruit_id = "<season>_<file stem>"` and `box = "<season>_<letter>"`. The 2425
    optimal-storage fruit (`G_opt`, `I_opt`) are put in the box of their letter; that they come
    from orchards G and I is not yet confirmed.
    """)
    return


@app.cell
def _(cfg):
    labels_2425 = read_label_file(cfg.path("labels_2425"), "2425")
    labels_2526 = read_label_file(cfg.path("labels_2526"), "2526")
    mo.vstack(
        [
            mo.md(
                f"2425: **{labels_2425.height}** fruit (660 expected). "
                f"2526: **{labels_2526.height}** fruit (450 expected)."
            ),
            mo.ui.table(labels_2425.head(5), selection=None, label="2425, first rows"),
            mo.ui.table(labels_2526.head(5), selection=None, label="2526, first rows"),
        ]
    )
    return labels_2425, labels_2526


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. Derived columns

    Hypothesis to test: `defective = max(browning, cavity)`, `binary_123 = defective ≥ 1`,
    `binary_23 = defective ≥ 2`. The table lists every fruit where a stored column differs from
    the derived one. Empty tables confirm the hypothesis.

    **Result (3 October 2026).** No disagreements in either season: all three rules hold for every
    fruit.
    """)
    return


@app.cell
def _(labels_2425, labels_2526):
    disagree_2425 = check_derived_labels(derive_labels(labels_2425))
    disagree_2526 = check_derived_labels(derive_labels(labels_2526))
    _ok = disagree_2425.is_empty() and disagree_2526.is_empty()
    mo.vstack(
        [
            mo.callout(
                mo.md(
                    f"Disagreements: 2425 **{disagree_2425.height}**, "
                    f"2526 **{disagree_2526.height}**."
                ),
                kind="success" if _ok else "warn",
            ),
            mo.ui.table(disagree_2425, selection=None, label="2425 disagreements"),
            mo.ui.table(disagree_2526, selection=None, label="2526 disagreements"),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `non-consumable` (2425 only) has no documented range or definition. The cross table against
    `defective` and `rot` may show how it was assigned. Ruben's hypothesis (3 October 2026): a
    fruit is non-consumable when `defective ≥ 2`, possibly also when `rot = 1`. The second table
    counts the fruit that contradict each candidate rule.

    **Result (3 October 2026).** `non-consumable = defective ≥ 2` holds for all 660 fruit. Adding
    `rot = 1` breaks the rule for one fruit (rot = 1, defective = 0, consumable), so rot plays no
    role in `non-consumable`.
    """)
    return


@app.cell
def _(labels_2425):
    _rules = {
        "defective >= 2": pl.col("defective") >= 2,
        "defective >= 2 or rot = 1": (pl.col("defective") >= 2) | (pl.col("rot") == 1),
    }
    mo.ui.table(
        pl.DataFrame(
            {
                "rule": list(_rules),
                "fruit contradicting the rule": [
                    labels_2425.filter(
                        _rule.cast(pl.Int64) != pl.col("non-consumable").cast(pl.Int64)
                    ).height
                    for _rule in _rules.values()
                ],
            }
        ),
        selection=None,
        label="2425: candidate rules for non-consumable",
    )
    return


@app.cell
def _(labels_2425):
    mo.ui.table(
        labels_2425.group_by("non-consumable", "defective", "rot")
        .len("n_fruit")
        .sort("non-consumable", "defective", "rot"),
        selection=None,
        label="2425: non-consumable against defective and rot",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3. Counts per box and per grade

    Both seasons combined on the columns they share (browning and cavity).
    """)
    return


@app.cell
def _(cfg):
    labels = load_labels(cfg)
    return (labels,)


@app.cell
def _(labels):
    _per_box = (
        labels.pivot(
            on="storage", index=["season", "box"], values="fruit_id", aggregate_function="len"
        )
        .fill_null(0)
        .with_columns(pl.sum_horizontal(pl.exclude("season", "box")).alias("total"))
        .sort("season", "box")
    )
    mo.ui.table(_per_box, selection=None, label="Fruit per box and storage group")
    return


@app.cell
def _(labels):
    def _grade_table(grade: str) -> pl.DataFrame:
        return (
            labels.sort(grade)
            .pivot(
                on=grade,
                index=["season", "storage"],
                values="fruit_id",
                aggregate_function="len",
            )
            .fill_null(0)
            .sort("season", "storage")
        )

    mo.vstack(
        [
            mo.ui.table(
                _grade_table("browning"),
                selection=None,
                label="Browning grade (columns) per season and storage group",
            ),
            mo.ui.table(
                _grade_table("cavity"),
                selection=None,
                label="Cavity grade (columns) per season and storage group, the control",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 4. Labelled fruit without a volume, and volumes without a label

    Only file names in the volume folders are read. The list of 17 fruit without a scan in 2526
    comes from the archive listing of 3 October 2026 (HANDOFF.md).

    **Result (3 October 2026).** Exactly those 17 labelled fruit have no volume; every volume has a
    label.
    """)
    return


@app.cell
def _(cfg):
    volumes = pl.concat(
        [
            list_volumes(cfg.path("ct_dir_2425"), "2425"),
            list_volumes(cfg.path("ct_dir_2526"), "2526"),
        ]
    )
    return (volumes,)


@app.cell
def _(labels, volumes):
    expected_missing = {
        f"2526_{_s}"
        for _s in (
            "C27 E24 F13 F22 F28 G08 G15 G16 G19 G23 G25 G28 H08 H24 I16 K25 N24".split()
        )
    }
    no_volume = labels.join(volumes, on="fruit_id", how="anti").sort("fruit_id")
    no_label = volumes.join(labels, on="fruit_id", how="anti").sort("fruit_id")
    _found = set(no_volume["fruit_id"])
    _as_expected = _found == expected_missing and no_label.is_empty()
    mo.vstack(
        [
            mo.callout(
                mo.md(
                    f"Volumes on disk: 2425 **{volumes.filter(pl.col('season') == '2425').height}**, "
                    f"2526 **{volumes.filter(pl.col('season') == '2526').height}**. "
                    f"Labelled fruit without a volume: **{no_volume.height}**. "
                    f"Volumes without a label: **{no_label.height}**. "
                    + (
                        "The missing fruit are exactly the 17 listed in HANDOFF.md."
                        if _as_expected
                        else f"Differs from HANDOFF.md: not expected "
                        f"{sorted(_found - expected_missing)}, expected but present "
                        f"{sorted(expected_missing - _found)}."
                    )
                ),
                kind="success" if _as_expected else "warn",
            ),
            mo.ui.table(no_volume, selection=None, label="Labelled fruit without a volume"),
            mo.ui.table(no_label, selection=None, label="Volumes without a label"),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 5. Rot and the highest grades

    Hugo (October 2026): rotten fruit were given the highest browning and cavity grades by hand.
    The 2526 file has no rot column, so rotten fruit there can only be suspected from
    browning 3 and cavity 3. In 2425 the `rot` column shows how far that suspicion holds.

    Questions: (a) in 2425, how many fruit with rot = 1 have browning 3 and cavity 3; (b) of the
    300 stored fruit with any defect, the 296 with browning and cavity together, how many are
    rot = 1; (c) in 2526, how many fruit per box have browning 3 and cavity 3.

    **Result (5 October 2026).** (a) 16 fruit in 2425 have rot = 1 (15 after suboptimal storage, 1
    after optimal storage, none at harvest); 11 of them have browning 3 and cavity 3. The other
    5 do not: one has browning 0 and cavity 0, the others have cavity 1 or 2. So "rotten fruit
    were given the highest grades" holds for 11 of 16 fruit in 2425, not for all. (b) Of the 296
    stored fruit with browning and cavity both at least 1, 15 are rot = 1; of the 175 fruit with
    browning 3 and cavity 3, 11 are rot = 1. Rot therefore explains only a small part of the
    browning–cavity overlap in 2425. (c) 2526: 24 fruit with browning 3 and cavity 3, at most 6 in
    one box (H). If 2526 resembles 2425, most of these 24 are not rotten, so flagging all of them
    removes real severe cases from the sensitivity analysis.
    """)
    return


@app.cell
def _(labels_2425):
    _b3c3 = (pl.col("browning") == 3) & (pl.col("cavity") == 3)
    _stored_defective = (pl.col("storage") != "harvest") & (
        pl.max_horizontal("browning", "cavity") >= 1
    )
    _both = (pl.col("browning") >= 1) & (pl.col("cavity") >= 1)
    rot_summary_2425 = pl.DataFrame(
        {
            "group": [
                "all fruit with rot = 1",
                "rot = 1 with browning 3 and cavity 3",
                "fruit with browning 3 and cavity 3",
                "stored fruit with any defect",
                "stored, browning and cavity both at least 1",
                "stored, both at least 1, rot = 1",
            ],
            "n_fruit": [
                labels_2425.filter(pl.col("rot") == 1).height,
                labels_2425.filter((pl.col("rot") == 1) & _b3c3).height,
                labels_2425.filter(_b3c3).height,
                labels_2425.filter(_stored_defective).height,
                labels_2425.filter(_stored_defective & _both).height,
                labels_2425.filter(
                    _stored_defective & _both & (pl.col("rot") == 1)
                ).height,
            ],
        }
    )
    mo.vstack(
        [
            mo.ui.table(rot_summary_2425, selection=None, label="2425: rot and grade 3"),
            mo.ui.table(
                labels_2425.group_by("rot", "browning", "cavity")
                .len("n_fruit")
                .sort("rot", "browning", "cavity"),
                selection=None,
                label="2425: rot by browning by cavity",
            ),
            mo.ui.table(
                labels_2425.group_by("storage", "rot").len("n_fruit").sort("storage", "rot"),
                selection=None,
                label="2425: rot per storage group",
            ),
        ]
    )
    return


@app.cell
def _(labels_2526):
    b3c3_per_box_2526 = (
        labels_2526.group_by("box")
        .agg(
            pl.len().alias("n_fruit"),
            (pl.col("browning") == 3).sum().alias("browning_3"),
            (pl.col("cavity") == 3).sum().alias("cavity_3"),
            ((pl.col("browning") == 3) & (pl.col("cavity") == 3))
            .sum()
            .alias("browning_3_and_cavity_3"),
        )
        .sort("box")
    )
    mo.vstack(
        [
            mo.md(
                "2526 fruit with browning 3 and cavity 3: "
                f"**{b3c3_per_box_2526['browning_3_and_cavity_3'].sum()}** "
                "(all 450 labelled fruit, scanned or not)."
            ),
            mo.ui.table(b3c3_per_box_2526, selection=None, label="2526 per box"),
        ]
    )
    return


if __name__ == "__main__":
    app.run()

"""The table of open fruit that the notebooks start from (28 October 2026).

Notebooks 05, 06 and 07 each built this table with their own code; it is built once here so the
grades, the cause of the defect and the faulty-volume exclusion are defined in one place.

One row per open fruit of the encoding run (`pearlatent.runs.ENCODING_RUN`): the manifest, with
`position` = the row of the fruit in `latents.npy` and `masks.npy`; the grades `browning` and
`cavity`; `defective` = the larger of the two; `cause` (charter rule 5); and the shape, stalk
and intensity descriptors (`pearlatent.descriptors`, run `DESCRIPTORS_RUN_DIR`).

Faulty volumes (`docs/outlier_rule.md`) are not dropped here: notebook 05 shows them, so it needs
them in the table. `drop_faulty_volumes` drops them where a notebook wants them gone.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from pearlatent.exclusions import read_volume_decisions

# The four causes of a defect, in the order the plots use (charter rule 5).
CAUSES = ("none", "browning only", "cavity only", "both")


def cause() -> pl.Expr:
    """Cause of the defect from the columns `browning` and `cavity`, named "cause".

    none: both 0; browning only: browning > 0 and cavity 0; cavity only: cavity > 0 and
    browning 0; both: both > 0.
    """
    return (
        pl.when((pl.col("browning") == 0) & (pl.col("cavity") == 0))
        .then(pl.lit(CAUSES[0]))
        .when(pl.col("cavity") == 0)
        .then(pl.lit(CAUSES[1]))
        .when(pl.col("browning") == 0)
        .then(pl.lit(CAUSES[2]))
        .otherwise(pl.lit(CAUSES[3]))
        .alias("cause")
    )


def attach_labels(
    manifest: pl.DataFrame, labels: pl.DataFrame, descriptors: pl.DataFrame
) -> pl.DataFrame:
    """The manifest with `position`, grades, `defective`, `cause` and the descriptors.

    In: the manifest of an encoding run (row n belongs to latent n), the label table (needs
    `fruit_id`, `browning`, `cavity`) and the descriptor table (one row per fruit). Out: one
    row per manifest row, in manifest order. Raises when a fruit has no grade or no
    descriptors, so no fruit is lost or left half-filled without notice.
    """
    out = (
        manifest.with_row_index("position")
        .join(
            labels.select("fruit_id", "browning", "cavity"),
            on="fruit_id",
            how="left",
            validate="1:1",
        )
        .join(descriptors, on="fruit_id", how="left", validate="1:1")
        .with_columns(defective=pl.max_horizontal("browning", "cavity"))
        .with_columns(cause())
    )
    no_grade = out.filter(pl.col("browning").is_null() | pl.col("cavity").is_null())
    if no_grade.height:
        raise ValueError(
            f"{no_grade.height} fruit without a grade: {no_grade['fruit_id'].to_list()[:5]}"
        )
    descriptor_column = next(c for c in descriptors.columns if c != "fruit_id")
    no_descriptors = out.filter(pl.col(descriptor_column).is_null())
    if no_descriptors.height:
        raise ValueError(
            f"{no_descriptors.height} fruit without descriptors:"
            f" {no_descriptors['fruit_id'].to_list()[:5]}"
        )
    return out


def open_fruit(cfg) -> pl.DataFrame:
    """`attach_labels` on the current encoding and descriptors runs (`pearlatent.runs`).

    In: the config. Out: one row per open fruit (940 on 28 October 2026), faulty volumes
    included.
    """
    from pearlatent.labels import load_labels
    from pearlatent.runs import DESCRIPTORS_RUN_DIR, ENCODING_RUN_DIR

    derived = cfg.path("derived_dir")
    manifest = pl.read_parquet(derived / ENCODING_RUN_DIR / "manifest.parquet")
    descriptors = pl.read_parquet(derived / DESCRIPTORS_RUN_DIR / "descriptors.parquet")
    return attach_labels(manifest, load_labels(cfg), descriptors)


def drop_faulty_volumes(fruit: pl.DataFrame, decisions_file: Path) -> pl.DataFrame:
    """`fruit` without the volumes Ruben judged faulty (`docs/outlier_rule.md`).

    In: a table with `fruit_id` and the path of `derived/volume_decisions.csv`. Out: the rows
    whose fruit has no decision other than "keep". When the file does not exist yet, `fruit`
    comes back unchanged; the caller prints how many fruit went in and came out.
    """
    if not Path(decisions_file).exists():
        return fruit
    faulty = read_volume_decisions(decisions_file).filter(pl.col("decision") != "keep")
    return fruit.join(faulty.select("fruit_id"), on="fruit_id", how="anti")

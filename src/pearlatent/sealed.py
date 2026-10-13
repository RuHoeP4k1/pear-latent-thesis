"""The sealed final-test fruit, and the one function every notebook uses to keep them out.

Charter rule 3 (13 October 2026): the final test has two sealed parts, reported separately:
(a) the 3 drawn 2526 test boxes (handoff B3; added when the draw is made);
(b) Hugo's 66 VAE test fruit of 2425, one batch, reported per storage group.
Sealed fruit stay out of exploration, plots and model choice until the pipeline is frozen.
`drop_sealed` removes them unless the caller passes `include_sealed=True` on purpose.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

SEALED_SCHEMA = {
    "fruit_id": pl.String,
    "season": pl.String,
    "sealed_part": pl.String,
}
PART_2425 = "2425 Hugo VAE test fruit"
PART_2526 = "2526 test boxes"


def build_sealed_fruit(
    hugo_split: pl.DataFrame, test_box_fruit: pl.DataFrame | None = None
) -> pl.DataFrame:
    """One row per sealed fruit: `fruit_id, season, sealed_part`.

    hugo_split: output of `pearlatent.hugo_split.hugo_vae_split_2425` (fruit_id, hugo_split);
        its "test" rows are sealed.
    test_box_fruit: the 2526 fruit of the drawn test boxes (fruit_id), or None before the draw.
    """
    parts = [
        hugo_split.filter(pl.col("hugo_split") == "test").select(
            "fruit_id",
            pl.lit("2425").alias("season"),
            pl.lit(PART_2425).alias("sealed_part"),
        )
    ]
    if test_box_fruit is not None:
        parts.append(
            test_box_fruit.select(
                "fruit_id",
                pl.lit("2526").alias("season"),
                pl.lit(PART_2526).alias("sealed_part"),
            )
        )
    out = pl.concat(parts).cast(SEALED_SCHEMA).sort("fruit_id")
    if out["fruit_id"].is_duplicated().any():
        raise ValueError("a fruit occurs twice in the sealed table")
    for part, n in out.group_by("sealed_part").len().iter_rows():
        print(f"sealed: {part}: {n} fruit")
    return out


def read_sealed_fruit(path: str | Path) -> pl.DataFrame:
    return pl.read_csv(path, schema=SEALED_SCHEMA)


def drop_sealed(
    df: pl.DataFrame, sealed: pl.DataFrame, include_sealed: bool = False
) -> pl.DataFrame:
    """Remove sealed fruit from `df` (needs `fruit_id`), and print how many went in and out.

    include_sealed=True keeps them; only the final evaluation, after the pipeline is frozen,
    may pass it.
    """
    if include_sealed:
        print(f"drop_sealed: include_sealed=True, all {df.height} fruit kept")
        return df
    out = df.join(sealed.select("fruit_id"), on="fruit_id", how="anti")
    print(
        f"drop_sealed: {df.height} fruit in, {df.height - out.height} sealed removed, "
        f"{out.height} out"
    )
    return out


def write_sealed_fruit(cfg) -> Path:
    """Write `derived/sealed_fruit.csv` from the files in `derived/` that define it."""
    derived = cfg.path("derived_dir")
    hugo_split = pl.read_csv(
        derived / "hugo_vae_split_2425" / "hugo_vae_split_2425.csv"
    )
    test_boxes_file = derived / "test_boxes" / "test_box_fruit.csv"
    test_box_fruit = pl.read_csv(test_boxes_file) if test_boxes_file.exists() else None
    if test_box_fruit is None:
        print("no 2526 test boxes drawn yet: only the 2425 part is sealed")
    out = derived / "sealed_fruit.csv"
    build_sealed_fruit(hugo_split, test_box_fruit).write_csv(out)
    print(f"written: {out}")
    return out


if __name__ == "__main__":
    from pearlatent.config import load_config

    write_sealed_fruit(load_config())

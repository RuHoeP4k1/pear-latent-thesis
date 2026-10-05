"""The table of excluded and flagged fruit, and the one function every notebook uses to apply it.

Rule 10 of CHARTER.md (5 October 2026):
- exclude: the 17 unscanned 2526 fruit (Hugo's advice: poor scan setup);
- flag: 2526 fruit with browning 3 and cavity 3 (possibly rotten; Hugo said rotten fruit were
  given the highest browning and cavity grades by hand, and the 2526 file has no rot column),
  until Hugo identifies the rotten fruit;
- flag: any volume flagged by the quality check (`pearlatent.volume_qc.flag_volumes`).

Mode "primary" drops excluded fruit; mode "sensitivity" also drops flagged fruit. A fruit may
have several rows, one per reason.
"""

from __future__ import annotations

import polars as pl

EXCLUSION_SCHEMA = {
    "fruit_id": pl.String,
    "season": pl.String,
    "reason": pl.String,
    "action": pl.String,
}
ACTIONS = ("exclude", "flag")
MODES = {"primary": ("exclude",), "sensitivity": ("exclude", "flag")}


def build_exclusions(
    labels: pl.DataFrame, volumes: pl.DataFrame, qc_flags: pl.DataFrame | None = None
) -> pl.DataFrame:
    """Build the exclusion table.

    labels: output of `load_labels` (needs fruit_id, season, browning, cavity).
    volumes: output of `list_volumes` for both seasons (needs fruit_id).
    qc_flags: output of `flag_volumes` (needs fruit_id, season, qc_flag), or None.
    """
    parts = [
        labels.join(volumes.select("fruit_id"), on="fruit_id", how="anti")
        .filter(pl.col("season") == "2526")
        .select(
            "fruit_id",
            "season",
            pl.lit("no CT volume (Hugo: poor scan setup)").alias("reason"),
            pl.lit("exclude").alias("action"),
        ),
        labels.filter(
            (pl.col("season") == "2526")
            & (pl.col("browning") == 3)
            & (pl.col("cavity") == 3)
        ).select(
            "fruit_id",
            "season",
            pl.lit(
                "browning 3 and cavity 3: possibly rotten, not yet identified"
            ).alias("reason"),
            pl.lit("flag").alias("action"),
        ),
    ]
    if qc_flags is not None:
        parts.append(
            qc_flags.filter(pl.col("qc_flag") != "").select(
                "fruit_id",
                "season",
                ("volume quality check: " + pl.col("qc_flag")).alias("reason"),
                pl.lit("flag").alias("action"),
            )
        )
    out = pl.concat(parts).cast(EXCLUSION_SCHEMA).sort("fruit_id", "action", "reason")
    _validate(out)
    return out


def _validate(exclusions: pl.DataFrame) -> None:
    if dict(exclusions.schema) != EXCLUSION_SCHEMA:
        raise ValueError(f"Exclusion table has columns {exclusions.schema}")
    bad = exclusions.filter(~pl.col("action").is_in(ACTIONS))
    if len(bad) > 0:
        raise ValueError(f"Unknown actions: {bad['action'].unique().to_list()}")


def read_exclusions(path) -> pl.DataFrame:
    out = pl.read_csv(path, schema=EXCLUSION_SCHEMA)
    _validate(out)
    return out


def apply_exclusions(
    df: pl.DataFrame, exclusions: pl.DataFrame, mode: str
) -> pl.DataFrame:
    """Drop the fruit that `mode` excludes. `df` needs a `fruit_id` column.

    mode "primary": drop fruit with action "exclude".
    mode "sensitivity": also drop fruit with action "flag".
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {list(MODES)}, got {mode!r}")
    _validate(exclusions)
    drop = (
        exclusions.filter(pl.col("action").is_in(MODES[mode]))
        .select("fruit_id")
        .unique()
    )
    return df.join(drop, on="fruit_id", how="anti")

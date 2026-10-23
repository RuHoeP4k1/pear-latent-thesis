"""The table of excluded and flagged fruit, and the one function every notebook uses to apply it.

Rule 10 of CHARTER.md (5 October 2026, flag reason updated 13 October 2026):
- exclude: the 17 unscanned 2526 fruit (Hugo's advice: poor scan setup);
- flag: 2526 fruit with browning 3 and cavity 3. Hugo (13 October 2026): rot is not really a
  problem in 2526, so these fruit stay in the primary analysis; the flag only drives the one
  sensitivity row without them (charter rule 10, version of 13 October);
- flag: any volume flagged by the quality check (`pearlatent.volume_qc.flag_volumes`);
- exclude: the 528 2425 fruit Hugo's VAE was trained on (charter rule 2, 13 October 2026; the
  encoder has seen them). They are used only in the one Step 0 comparison of reconstruction
  error, selected there on purpose through `HUGO_TRAINING_REASON`;
- exclude: faulty volumes, decided by Ruben fruit by fruit under `docs/outlier_rule.md`
  (23 October 2026) and read from `derived/volume_decisions.csv`.

Mode "primary" drops excluded fruit; mode "sensitivity" also drops flagged fruit; mode
"all_volumes" drops excluded fruit except the faulty volumes (the sensitivity row of
`docs/outlier_rule.md`, section 5). A fruit may have several rows, one per reason.
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
MODES = {
    "primary": ("exclude",),
    "sensitivity": ("exclude", "flag"),
    "all_volumes": ("exclude",),
}
HUGO_TRAINING_REASON = "Hugo VAE training fruit: Step 0 comparison only"

# Faulty volumes (docs/outlier_rule.md). Every reason for such a row starts with this text, so
# mode "all_volumes" can keep these fruit.
FAULTY_VOLUME_PREFIX = "faulty volume: "
VOLUME_DECISION_SCHEMA = {
    "fruit_id": pl.String,
    "season": pl.String,
    "decision": pl.String,
    "note": pl.String,
}
# missing_part and preprocessing exclude the fruit; keep records that a candidate was judged.
VOLUME_DECISIONS = ("missing_part", "preprocessing", "keep")


def build_exclusions(
    labels: pl.DataFrame,
    volumes: pl.DataFrame,
    qc_flags: pl.DataFrame | None = None,
    hugo_split: pl.DataFrame | None = None,
    volume_decisions: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Build the exclusion table.

    labels: output of `load_labels` (needs fruit_id, season, browning, cavity).
    volumes: output of `list_volumes` for both seasons (needs fruit_id).
    qc_flags: output of `flag_volumes` (needs fruit_id, season, qc_flag), or None.
    hugo_split: output of `pearlatent.hugo_split.hugo_vae_split_2425` (needs fruit_id,
        hugo_split), or None.
    volume_decisions: output of `read_volume_decisions`, or None. Fruit with decision
        missing_part or preprocessing get action "exclude"; "keep" adds no row.
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
                "browning 3 and cavity 3: sensitivity row only (charter rule 10)"
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
    if hugo_split is not None:
        parts.append(
            hugo_split.filter(pl.col("hugo_split") == "train").select(
                "fruit_id",
                pl.lit("2425").alias("season"),
                pl.lit(HUGO_TRAINING_REASON).alias("reason"),
                pl.lit("exclude").alias("action"),
            )
        )
    if volume_decisions is not None:
        parts.append(
            volume_decisions.filter(pl.col("decision") != "keep").select(
                "fruit_id",
                "season",
                (pl.lit(FAULTY_VOLUME_PREFIX) + pl.col("decision")).alias("reason"),
                pl.lit("exclude").alias("action"),
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


def read_volume_decisions(path) -> pl.DataFrame:
    """Read Ruben's decisions on candidate volumes (`docs/outlier_rule.md`, section 6).

    In: a CSV with columns fruit_id, season, decision, note. Out: the same table, checked:
    every decision is one of `VOLUME_DECISIONS` and every fruit occurs once.
    """
    out = pl.read_csv(path, schema=VOLUME_DECISION_SCHEMA).with_columns(
        pl.col("note").fill_null("")
    )
    unknown = out.filter(~pl.col("decision").is_in(VOLUME_DECISIONS))
    if unknown.height:
        raise ValueError(
            f"Unknown decisions {unknown['decision'].unique().to_list()};"
            f" allowed: {VOLUME_DECISIONS}"
        )
    twice = out.filter(pl.col("fruit_id").is_duplicated())
    if twice.height:
        raise ValueError(
            f"Fruit decided more than once: {twice['fruit_id'].unique().to_list()}"
        )
    return out


def apply_exclusions(
    df: pl.DataFrame, exclusions: pl.DataFrame, mode: str
) -> pl.DataFrame:
    """Drop the fruit that `mode` excludes. `df` needs a `fruit_id` column.

    mode "primary": drop fruit with action "exclude".
    mode "sensitivity": also drop fruit with action "flag".
    mode "all_volumes": drop fruit with action "exclude", but keep the faulty volumes.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {list(MODES)}, got {mode!r}")
    _validate(exclusions)
    rows = exclusions.filter(pl.col("action").is_in(MODES[mode]))
    if mode == "all_volumes":
        rows = rows.filter(~pl.col("reason").str.starts_with(FAULTY_VOLUME_PREFIX))
    drop = rows.select("fruit_id").unique()
    return df.join(drop, on="fruit_id", how="anti")

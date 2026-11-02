"""The fixed-feature levels of charter question 3, loaded the same way for every model.

Each feature set gives one row of 32,768 numbers per fruit:

- `hugo_vae`: `mu` of Hugo's encoder (encoding run `20261013T171306Z_hugo_vae`, open part);
- `random_init`: `mu` of the same architecture with random weights, seed 0 (run
  `20261020T120336Z_random_init`, open part); separates the training from the architecture;
- `ct_32`: the CT volume averaged in blocks of 4³ voxels (run `20261005T142026Z_raw_ct_32`);
  places a loss in the resolution;
- `tissue_features`: the 10 global intensity features of Van De Looverbosch et al. (2020) on the
  128³ volumes (`pearlatent.tissue_features`, run `20261102T151947Z_tissue_features`, open
  fruit only). Ten numbers per fruit in different units, so models scale them (inside the
  training fold).

The CT run of 5 October was made before the test boxes were drawn and holds all 1,093
volumes, the sealed fruit included. `load_features` therefore always removes the sealed fruit
(`pearlatent.sealed.drop_sealed`) unless `include_sealed=True`, which only the final test may
pass. The 128³ input level (2,097,152 numbers per fruit) is not here yet.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from pearlatent.runs import (
    ENCODING_RUN_DIR,
    RANDOM_INIT_RUN_DIR,
    RAW_CT_32_RUN_DIR,
    TISSUE_FEATURES_RUN_DIR,
)
from pearlatent.sealed import drop_sealed, read_sealed_fruit
from pearlatent.tissue_features import FEATURE_NAMES

FEATURE_RUNS = {
    "hugo_vae": (ENCODING_RUN_DIR, "latents.npy"),
    "random_init": (RANDOM_INIT_RUN_DIR, "latents.npy"),
    "ct_32": (RAW_CT_32_RUN_DIR, "grids.npy"),
}


def load_features(
    cfg, name: str, fruit_ids: list[str], include_sealed: bool = False
) -> np.ndarray:
    """Features of `fruit_ids`, in that order, as an array (fruit, 32,768; 10 for
    `tissue_features`).

    In: the config, the feature set name, the fruit wanted. Out: float32 rows. Raises when a
    fruit is not in the feature set or when a sealed fruit is asked for without
    `include_sealed`.
    """
    if name == "tissue_features":
        return _load_tissue_features(cfg, fruit_ids, include_sealed)
    if name not in FEATURE_RUNS:
        known = sorted([*FEATURE_RUNS, "tissue_features"])
        raise ValueError(f"unknown feature set {name!r}; known: {known}")
    folder, array_file = FEATURE_RUNS[name]
    run = cfg.path("derived_dir") / folder
    manifest = pl.read_parquet(run / "manifest.parquet").with_row_index("position")
    sealed = read_sealed_fruit(cfg.path("derived_dir") / "sealed_fruit.csv")
    available = drop_sealed(manifest, sealed, include_sealed=include_sealed)

    wanted = pl.DataFrame({"fruit_id": fruit_ids}).with_row_index("order")
    found = wanted.join(
        available.select("fruit_id", "position"), on="fruit_id", how="left"
    )
    missing = found.filter(pl.col("position").is_null())["fruit_id"].to_list()
    if missing:
        raise ValueError(
            f"{len(missing)} fruit not available in {name} (sealed or absent): {missing[:5]}"
        )
    positions = found.sort("order")["position"].to_numpy()
    array = np.load(run / array_file, mmap_mode="r")
    return np.asarray(array[positions], dtype=np.float32).reshape(len(positions), -1)


def _load_tissue_features(
    cfg, fruit_ids: list[str], include_sealed: bool
) -> np.ndarray:
    """The 10 columns of `FEATURE_NAMES` for `fruit_ids`, in that order (same checks)."""
    table = pl.read_parquet(
        cfg.path("derived_dir") / TISSUE_FEATURES_RUN_DIR / "tissue_features.parquet"
    )
    sealed = read_sealed_fruit(cfg.path("derived_dir") / "sealed_fruit.csv")
    available = drop_sealed(table, sealed, include_sealed=include_sealed)
    wanted = pl.DataFrame({"fruit_id": fruit_ids}).with_row_index("order")
    found = wanted.join(available, on="fruit_id", how="left", validate="1:1")
    missing = found.filter(pl.col(FEATURE_NAMES[0]).is_null())["fruit_id"].to_list()
    if missing:
        raise ValueError(
            f"{len(missing)} fruit not available in tissue_features: {missing[:5]}"
        )
    return found.sort("order").select(FEATURE_NAMES).to_numpy().astype(np.float32)

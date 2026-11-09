"""Counterfactual images (charter, Extensions; notebook 17).

Plan (Ruben, 7 and 9 November 2026): move a fruit's latent `mu` along the direction of a PLS-DA
model on the full 32³ latent grid and decode the result, for two directions on the same fruit:
the cavity-free browning model and the general `binary_23` model. First step, here: which fruit.

`draw_example_fruit`: 2 grade 3 and 2 grade 0 fruit, all cavity-free, from 4 different boxes,
with a fixed seed. Cavity-free so that both directions apply to the same fruit and a change in
the image cannot come from an existing cavity. Ruben looks at their CT slices before any decode.
"""

from __future__ import annotations

import numpy as np
import polars as pl

DRAW_SEED = 369
N_PER_GROUP = 2


def draw_example_fruit(
    fruit: pl.DataFrame, n_per_group: int = N_PER_GROUP, seed: int = DRAW_SEED
) -> pl.DataFrame:
    """The example fruit: `n_per_group` of grade 3 and of grade 0, every fruit from its own box.

    In: a table with `fruit_id`, `box`, `browning`, `cavity`, `defective` (the open fruit after the
    faulty volumes). Out: the drawn rows with a `group` column ("grade 3" first, then "grade 0").

    Grade 3 = browning 3 and cavity 0; grade 0 = defective 0. Each candidate list is sorted by
    fruit_id (so the row order of `fruit` does not matter) and shuffled with one generator
    `default_rng(seed)`; walking down the shuffled list, a fruit is taken when its box is not used
    yet. Grade 3 first, because it has fewer candidates and boxes.
    """
    rng = np.random.default_rng(seed)
    groups = {
        "grade 3": (pl.col("browning") == 3) & (pl.col("cavity") == 0),
        "grade 0": pl.col("defective") == 0,
    }
    used_boxes: set[str] = set()
    picked = []
    for group, condition in groups.items():
        candidates = fruit.filter(condition).sort("fruit_id")
        print(
            f"{group}: {candidates.height} candidates in {candidates['box'].n_unique()} boxes"
        )
        taken = 0
        for index in rng.permutation(candidates.height):
            row = candidates.row(int(index), named=True)
            if row["box"] in used_boxes:
                continue
            picked.append({**row, "group": group})
            used_boxes.add(row["box"])
            taken += 1
            if taken == n_per_group:
                break
        if taken < n_per_group:
            raise ValueError(f"{group}: not enough boxes left for {n_per_group} fruit")
    return pl.DataFrame(picked).select("group", *fruit.columns)

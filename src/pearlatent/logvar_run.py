"""Encode the encoder's own uncertainty grid, `logvar`, for chosen open fruit (Ruben, 9 Nov 2026).

Hugo's encoder gives two 32³ grids per fruit: `mu` (the latent we analyse, saved in the encoding
run) and `logvar`, the log of the variance of the latent cell. The encoding run of 13 October
saved only `mu`. With a KL weight of 1e-6 the SD, exp(logvar / 2), is tiny (about 0.001 inside the
fruit against a spread of `mu` of about 5), so it cannot scale `mu` (no z-score). It may still
carry information: in a scratch sample of 12 fruit it was higher in browning 3 fruit than in
grade 0 fruit. This run saves it so that can be tested on all fruit.

The same call as the encoding run: min-max scaled volume, `vae.encoder(x)` on the CPU, no
augmentation. `mu` is computed again on the way and compared with the saved `mu`: the largest
absolute difference per fruit is stored, as a check that it is the same encoder and input.

    uv run python -m pearlatent.logvar_run 2526 validation

Arguments: which fruit, any of `2526` (open 2526 fruit) and `validation` (Hugo's 66 2425
validation fruit). Writes `derived/logvar/<timestamp>_logvar/logvar.npy` (fruit, 32, 32, 32,
float32), `manifest.parquet` (`fruit_id`, row, largest `mu` difference) and `run.json`.
"""

from __future__ import annotations

import sys
import time

import numpy as np

from pearlatent.runs import ENCODING_RUN, ENCODING_RUN_DIR


def main(groups: list[str]) -> None:
    import nibabel as nib
    import polars as pl
    import torch

    from pearlatent.config import load_config
    from pearlatent.encoding import load_vae
    from pearlatent.pooling import scale_minmax
    from pearlatent.runs import new_run_dir, write_run_record

    known = {"2526", "validation"}
    if not groups or set(groups) - known:
        raise ValueError(f"name one or more of {sorted(known)}")
    cfg = load_config()
    run = cfg.path("derived_dir") / ENCODING_RUN_DIR
    manifest = pl.read_parquet(run / "manifest.parquet").with_row_index("position")
    wanted = pl.lit(False)
    if "2526" in groups:
        wanted = wanted | (pl.col("season") == "2526")
    if "validation" in groups:
        split = pl.read_csv(
            cfg.path("derived_dir") / "hugo_vae_split_2425" / "hugo_vae_split_2425.csv"
        )
        validation = split.filter(pl.col("hugo_split") == "validation")["fruit_id"]
        wanted = wanted | pl.col("fruit_id").is_in(validation.to_list())
    chosen = manifest.filter(wanted)
    print(f"{manifest.height} open fruit in the encoding run, {chosen.height} chosen")

    saved_mu = np.load(run / "latents.npy", mmap_mode="r")
    # CPU, as the encoding run, so `mu` can be compared with the saved one.
    vae, info = load_vae(cfg.path("hugo_repo"), cfg.path("encoder_dir"), "cpu")
    logvars = np.empty((chosen.height, 32, 32, 32), dtype=np.float32)
    mu_differences = np.empty(chosen.height)
    start = time.time()
    for row, fruit in enumerate(chosen.iter_rows(named=True)):
        path = cfg.path(f"ct_dir_{fruit['season']}") / f"{fruit['file_stem']}.nii"
        volume = scale_minmax(np.asarray(nib.load(path).dataobj))
        x = torch.from_numpy(np.ascontiguousarray(volume, dtype=np.float32))[None, None]
        with torch.inference_mode():
            mu, logvar = vae.encoder(x)
        logvars[row] = logvar[0, 0].numpy()
        mu_differences[row] = np.abs(
            mu[0, 0].numpy() - saved_mu[fruit["position"]]
        ).max()
        if (row + 1) % 50 == 0:
            print(
                f"{row + 1} of {chosen.height}, {time.time() - start:.0f} s", flush=True
            )

    out_dir = new_run_dir(cfg.path("derived_dir") / "logvar", "logvar")
    np.save(out_dir / "logvar.npy", logvars)
    chosen.select("fruit_id").with_row_index("row").with_columns(
        largest_mu_difference=pl.Series(mu_differences)
    ).write_parquet(out_dir / "manifest.parquet")
    write_run_record(
        out_dir,
        {
            "what": "logvar of Hugo's encoder, same input and device as the encoding run",
            "encoding_run": ENCODING_RUN,
            "groups": groups,
            "n_fruit": chosen.height,
            "largest_mu_difference": float(mu_differences.max()),
            **info,
        },
        root=cfg.root,
    )
    print(
        f"wrote {chosen.height} fruit to {out_dir} in {time.time() - start:.0f} s;"
        f" largest mu difference {mu_differences.max():.1e}"
    )


if __name__ == "__main__":
    main(sys.argv[1:])

"""Decode a few fruit in full, for the 3D slice viewer and the GIF of notebook 06 (section 7).

`planes.npy` keeps only three planes per fruit; scrolling through a whole fruit needs the full
reconstruction. Decoding takes about 56 s per fruit on the laptop CPU (13 October 2026), so
only a handful of example fruit are decoded. The latent is the saved `mu` of the encoding run
(not re-encoded), decoded with `vae.decode(mu)`, the same call as the run itself.

    uv run python -m pearlatent.decode_examples 2526_A27 2526_B15 ...

Writes `derived/step0_examples/<timestamp>_examples/volumes.npz` (CT and reconstruction as
float16, about 8 MB per fruit) and `run.json`.
"""

from __future__ import annotations

import sys

import numpy as np

ENCODING_RUN = "20261013T171306Z_hugo_vae"


def main(fruit_ids: list[str]) -> None:
    import nibabel as nib
    import polars as pl
    import torch

    from pearlatent.config import load_config
    from pearlatent.device import get_device
    from pearlatent.encoding import load_vae
    from pearlatent.pooling import scale_minmax
    from pearlatent.runs import new_run_dir, write_run_record

    cfg = load_config()
    run = cfg.path("derived_dir") / "latents" / "hugo_vae" / ENCODING_RUN
    manifest = pl.read_parquet(run / "manifest.parquet").with_row_index("position")
    chosen = manifest.filter(pl.col("fruit_id").is_in(fruit_ids))
    missing = set(fruit_ids) - set(chosen["fruit_id"].to_list())
    if missing:
        raise ValueError(
            f"not among the open fruit of {ENCODING_RUN}: {sorted(missing)}"
        )
    latents = np.load(run / "latents.npy", mmap_mode="r")
    # CPU, as for the encoding run, so the reconstruction matches `planes.npy`.
    device = get_device(cfg.compute("device", "cpu"))
    vae, info = load_vae(cfg.path("hugo_repo"), cfg.path("encoder_dir"), device)

    ct_volumes, reconstructions = [], []
    for fruit in chosen.iter_rows(named=True):
        print(f"decoding {fruit['fruit_id']} ...", flush=True)
        path = cfg.path(f"ct_dir_{fruit['season']}") / f"{fruit['file_stem']}.nii"
        ct_volumes.append(scale_minmax(np.asarray(nib.load(path).dataobj)))
        mu = torch.from_numpy(np.ascontiguousarray(latents[fruit["position"]]))
        with torch.inference_mode():
            x_hat = vae.decode(mu.reshape(1, 1, 32, 32, 32).to(device)).float().cpu()
        reconstructions.append(x_hat[0, 0].numpy())

    out_dir = new_run_dir(cfg.path("derived_dir") / "step0_examples", "examples")
    np.savez_compressed(
        out_dir / "volumes.npz",
        fruit_id=np.array(chosen["fruit_id"].to_list()),
        ct=np.stack(ct_volumes).astype(np.float16),
        reconstruction=np.stack(reconstructions).astype(np.float16),
    )
    write_run_record(
        out_dir,
        {"encoding_run": ENCODING_RUN, "fruit": chosen["fruit_id"].to_list(), **info},
    )
    print(f"wrote {chosen.height} fruit to {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1:])

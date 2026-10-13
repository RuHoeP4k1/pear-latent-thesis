"""One pass over a group of fruit: encode, and with Hugo's decoder also reconstruct.

Called by notebook 04, once for the open fruit and once for the sealed fruit (each to its own
folder). Per fruit it writes the latent `mu`, the pooled 512-channel layer and the 32³ body mask
(as notebook 04 did before 13 October), and, when `reconstruct` is true, the reconstruction
numbers of `pearlatent.reconstruction` for `x_hat = vae.decode(mu)`.

Files in `out_dir`:
- `latents.npy` (n, 32, 32, 32), `layer512_pooled.npy` (n, 2, 512), `masks.npy` (n, 32, 32, 32);
- `manifest.parquet` (the rows of `volumes` without the path, in file order of the arrays);
- `features.parquet` (`mu` pooled with `pearlatent.pooling.pool_grids`);
- with `reconstruct`: `reconstruction.parquet` (one row per fruit), `slice_mae_volume.npy` and
  `slice_mae_body.npy` (n, 128: mean absolute residual per slice along the stem-calyx axis, over
  the whole slice and over body voxels only), `planes.npy` (n, 2, 3, 128, 128: original and
  reconstruction in the three planes through the body centre; float32, about 0.4 MB per fruit,
  so figures never need a second decoding pass).
"""

from __future__ import annotations

import time
from pathlib import Path

import nibabel as nib
import numpy as np
import polars as pl

from pearlatent.encoding import encode_volumes
from pearlatent.pooling import body_mask, fruit_mask, pool_grids, scale_minmax
from pearlatent.reconstruction import (
    body_centre,
    body_voxel_mask,
    central_planes,
    core_voxel_mask,
    masked_mae,
    slice_mae,
    whole_volume_metrics,
)


def reconstruct_one(vae, scaled: np.ndarray, mu: np.ndarray, body_cells, device: str):
    """Decode one fruit's `mu` and reduce the result to numbers, slice profiles and planes.

    In: the scaled volume (128³), its `mu` (32³) and its 32³ body mask.
    Out: a dict of per-fruit numbers, two slice profiles (128 each) and the planes (2, 3, 128, 128).
    """
    import torch

    with torch.inference_mode():
        mu_t = torch.from_numpy(np.ascontiguousarray(mu)).reshape(1, 1, *mu.shape)
        x_hat_t = vae.decode(mu_t.to(device)).float().cpu()
    x_t = torch.from_numpy(np.ascontiguousarray(scaled)).reshape(1, 1, *scaled.shape)
    numbers = whole_volume_metrics(x_t, x_hat_t)

    x_hat = x_hat_t[0, 0].numpy()
    abs_residual = np.abs(x_hat - scaled)
    body = body_voxel_mask(scaled, body_cells)
    core, geometry = core_voxel_mask(body)
    numbers |= {
        "mae_body": masked_mae(abs_residual, body),
        "mae_core": masked_mae(abs_residual, core),
        "n_body_voxels": int(body.sum()),
        "n_core_voxels": int(core.sum()),
        **geometry,
    }
    centre = body_centre(body)
    numbers |= {"plane_x": centre[0], "plane_y": centre[1], "plane_z": centre[2]}
    profiles = (slice_mae(abs_residual), slice_mae(abs_residual, body))
    planes = np.stack([central_planes(scaled, centre), central_planes(x_hat, centre)])
    return numbers, profiles, planes


def encode_group(
    vae,
    volumes: pl.DataFrame,
    out_dir: Path,
    device: str,
    batch_size: int,
    reconstruct: bool,
    bf16: bool = False,
    progress=lambda batches: batches,
) -> float:
    """Encode (and reconstruct) every row of `volumes` into `out_dir`; return seconds per fruit.

    `volumes` needs `fruit_id` and `path`; every other column is copied to the manifest.
    """
    n = volumes.height
    print(f"{out_dir.name}: {n} fruit, reconstruct={reconstruct}")
    base_channel = vae.encoder.feature_layers[0].out_channels
    mu_all = np.lib.format.open_memmap(
        out_dir / "latents.npy", mode="w+", dtype=np.float32, shape=(n, 32, 32, 32)
    )
    pooled_all = np.lib.format.open_memmap(
        out_dir / "layer512_pooled.npy",
        mode="w+",
        dtype=np.float32,
        shape=(n, 2, 2 * base_channel),
    )
    masks = np.zeros((n, 32, 32, 32), dtype=bool)
    if reconstruct:
        planes_all = np.lib.format.open_memmap(
            out_dir / "planes.npy",
            mode="w+",
            dtype=np.float32,
            shape=(n, 2, 3, 128, 128),
        )
        slice_volume = np.full((n, 128), np.nan, dtype=np.float32)
        slice_body = np.full((n, 128), np.nan, dtype=np.float32)
        rows = []
    paths = volumes["path"].to_list()
    start = time.perf_counter()
    for i in progress(list(range(0, n, batch_size))):
        j = min(i + batch_size, n)
        scaled = np.stack(
            [scale_minmax(np.asarray(nib.load(p).dataobj)) for p in paths[i:j]]
        )
        masks[i:j] = [body_mask(fruit_mask(s)) for s in scaled]
        mu_all[i:j], pooled_all[i:j] = encode_volumes(
            vae.encoder, scaled, masks[i:j], device, bf16=bf16
        )
        if reconstruct:
            for b in range(j - i):
                numbers, profiles, planes = reconstruct_one(
                    vae, scaled[b], mu_all[i + b], masks[i + b], device
                )
                rows.append({"fruit_id": volumes["fruit_id"][i + b], **numbers})
                slice_volume[i + b], slice_body[i + b] = profiles
                planes_all[i + b] = planes
    mu_all.flush()
    pooled_all.flush()
    np.save(out_dir / "masks.npy", masks)
    volumes.drop("path").write_parquet(out_dir / "manifest.parquet")
    pool_grids(np.asarray(mu_all), masks, volumes["fruit_id"].to_list()).write_parquet(
        out_dir / "features.parquet"
    )
    if reconstruct:
        planes_all.flush()
        np.save(out_dir / "slice_mae_volume.npy", slice_volume)
        np.save(out_dir / "slice_mae_body.npy", slice_body)
        pl.DataFrame(rows).write_parquet(out_dir / "reconstruction.parquet")
    seconds_per_fruit = (time.perf_counter() - start) / max(n, 1)
    print(f"{out_dir.name}: done, {seconds_per_fruit:.1f} s per fruit")
    return seconds_per_fruit

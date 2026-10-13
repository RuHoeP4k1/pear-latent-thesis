"""Encode a group of fruit, and reconstruct the ones marked for decoding, in resumable chunks.

Called by notebook 04, once for the open fruit and once for the sealed fruit (each to its own
folder). Decoding is slow on the laptop CPU (about 56 s per fruit, against about 4 s to encode;
measured 13 October 2026), so only the rows with `decode = True` are reconstructed, and the
work is cut into chunks of `CHUNK_SIZE` fruit that are saved one by one. A run stopped by
Ctrl-C, by `deadline`, or by a crash loses at most the chunk it was working on; the same call
with the same `out_dir` continues with the first missing chunk.

While working, in `out_dir`:
- `plan.parquet`: the rows of `volumes` without the path, written once; a resumed run refuses
  to continue when its rows differ;
- `parts/chunk_NNNN.npz` and `parts/chunk_NNNN.parquet`: one finished chunk. Both are written
  to a temporary name first and then renamed, and the `.npz` is written last, so a chunk counts
  as finished only when its `.npz` exists.

After the last chunk, `assemble` writes the final files:
- `latents.npy` (n, 32, 32, 32), `layer512_pooled.npy` (n, 2, 512), `masks.npy` (n, 32, 32, 32);
- `manifest.parquet` (plan rows plus `recon_row`: the row of the fruit in the reconstruction
  files, null when it was not decoded);
- `features.parquet` (`mu` pooled with `pearlatent.pooling.pool_grids`);
- for the decoded fruit, in `recon_row` order: `reconstruction.parquet`, `slice_mae_volume.npy`
  and `slice_mae_body.npy` (d, 128: mean absolute residual per slice along the stem-calyx axis,
  over the whole slice and over body voxels only), `planes.npy` (d, 2, 3, 128, 128: original and
  reconstruction in the three planes through the body centre, float32, about 0.4 MB per fruit,
  so figures never need a second decoding pass).
"""

from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager
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

# 20 fruit: about 20 minutes when all are decoded, so a stop loses little work.
CHUNK_SIZE = 20


@contextmanager
def keep_awake():
    """Ask Windows not to sleep while the block runs (no effect on other systems).

    Uses SetThreadExecutionState, the standard call for a long-running program; it changes no
    power setting and ends when the block ends or the process stops.
    """
    if sys.platform != "win32":
        yield
        return
    import ctypes

    es_continuous, es_system_required = 0x80000000, 0x00000001
    kernel32 = ctypes.windll.kernel32
    kernel32.SetThreadExecutionState(es_continuous | es_system_required)
    try:
        yield
    finally:
        kernel32.SetThreadExecutionState(es_continuous)


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


def chunk_bounds(n: int, chunk_size: int = CHUNK_SIZE) -> list[tuple[int, int]]:
    """(start, stop) row ranges of the chunks: [(0, 20), (20, 40), ...]."""
    return [(a, min(a + chunk_size, n)) for a in range(0, n, chunk_size)]


def _chunk_paths(out_dir: Path, k: int) -> tuple[Path, Path]:
    parts = out_dir / "parts"
    return parts / f"chunk_{k:04d}.npz", parts / f"chunk_{k:04d}.parquet"


def _write_atomic(path: Path, write) -> None:
    """Write through `write(temporary_path)`, then rename, so `path` is never half written."""
    tmp = path.with_name(path.stem + ".tmp" + path.suffix)
    write(tmp)
    os.replace(tmp, path)


def check_plan(volumes: pl.DataFrame, out_dir: Path) -> None:
    """Write `plan.parquet` on the first call; on a resumed run, refuse when the rows differ."""
    plan = volumes.drop("path")
    plan_file = out_dir / "plan.parquet"
    if not plan_file.exists():
        _write_atomic(plan_file, plan.write_parquet)
        return
    if not pl.read_parquet(plan_file).equals(plan):
        raise RuntimeError(
            f"{plan_file} lists other fruit or flags than this run; a resumed run must use "
            "the same volumes, limit and decode flags"
        )


def run_chunk(vae, chunk: pl.DataFrame, device: str, batch_size: int, bf16: bool):
    """Encode every fruit of `chunk`, and reconstruct the rows with `decode = True`.

    Out: a dict of arrays for the `.npz` file and the table of reconstruction numbers.
    """
    n = chunk.height
    mu = np.zeros((n, 32, 32, 32), dtype=np.float32)
    # Sized from the first batch: (n, 2, channels of the layer before the latent head).
    pooled = None
    masks = np.zeros((n, 32, 32, 32), dtype=bool)
    decode = chunk["decode"].to_list()
    paths = chunk["path"].to_list()
    planes, slice_volume, slice_body, rows = [], [], [], []
    for i in range(0, n, batch_size):
        j = min(i + batch_size, n)
        scaled = np.stack(
            [scale_minmax(np.asarray(nib.load(p).dataobj)) for p in paths[i:j]]
        )
        masks[i:j] = [body_mask(fruit_mask(s)) for s in scaled]
        mu[i:j], pooled_batch = encode_volumes(
            vae.encoder, scaled, masks[i:j], device, bf16=bf16
        )
        if pooled is None:
            pooled = np.zeros((n, *pooled_batch.shape[1:]), dtype=np.float32)
        pooled[i:j] = pooled_batch
        for b in range(j - i):
            if not decode[i + b]:
                continue
            numbers, profiles, fruit_planes = reconstruct_one(
                vae, scaled[b], mu[i + b], masks[i + b], device
            )
            rows.append({"fruit_id": chunk["fruit_id"][i + b], **numbers})
            slice_volume.append(profiles[0])
            slice_body.append(profiles[1])
            planes.append(fruit_planes)
    arrays = {
        "fruit_id": np.array(chunk["fruit_id"].to_list()),
        "mu": mu,
        "pooled": pooled,
        "masks": masks,
        "planes": np.array(planes, dtype=np.float32).reshape(-1, 2, 3, 128, 128),
        "slice_volume": np.array(slice_volume, dtype=np.float32).reshape(-1, 128),
        "slice_body": np.array(slice_body, dtype=np.float32).reshape(-1, 128),
    }
    return arrays, pl.DataFrame(rows)


def encode_in_chunks(
    vae,
    volumes: pl.DataFrame,
    out_dir: Path,
    device: str,
    batch_size: int,
    bf16: bool = False,
    chunk_size: int = CHUNK_SIZE,
    deadline: float | None = None,
    max_chunks: int | None = None,
) -> bool:
    """Work through the missing chunks of `volumes`; return True when every chunk is finished.

    `volumes` needs `fruit_id`, `path` and `decode`. Stops before starting a chunk once
    `time.time()` has passed `deadline`, or after `max_chunks` new chunks (used by the tests).
    """
    check_plan(volumes, out_dir)
    (out_dir / "parts").mkdir(exist_ok=True)
    bounds = chunk_bounds(volumes.height, chunk_size)
    done = [k for k in range(len(bounds)) if _chunk_paths(out_dir, k)[0].exists()]
    print(
        f"{out_dir.name}: {volumes.height} fruit ({int(volumes['decode'].sum())} to decode) "
        f"in {len(bounds)} chunks; {len(done)} already finished"
    )
    new = 0
    for k, (a, b) in enumerate(bounds):
        npz_path, table_path = _chunk_paths(out_dir, k)
        if npz_path.exists():
            continue
        if deadline is not None and time.time() >= deadline:
            print(f"{out_dir.name}: stop time reached before chunk {k}")
            return False
        if max_chunks is not None and new >= max_chunks:
            return False
        start = time.perf_counter()
        arrays, table = run_chunk(
            vae, volumes.slice(a, b - a), device, batch_size, bf16
        )
        _write_atomic(table_path, table.write_parquet)
        _write_atomic(npz_path, lambda p, arr=arrays: np.savez(p, **arr))
        new += 1
        print(
            f"{out_dir.name}: chunk {k + 1}/{len(bounds)} saved ({b - a} fruit, "
            f"{table.height} decoded, {time.perf_counter() - start:.0f} s)"
        )
    return True


def assemble(
    volumes: pl.DataFrame, out_dir: Path, chunk_size: int = CHUNK_SIZE
) -> None:
    """Combine the finished chunks into the final files listed in the module docstring."""
    bounds = chunk_bounds(volumes.height, chunk_size)
    chunks, tables = [], []
    for k, (a, b) in enumerate(bounds):
        npz_path, table_path = _chunk_paths(out_dir, k)
        with np.load(npz_path) as data:
            chunk = {name: data[name] for name in data.files}
        expected = volumes["fruit_id"].slice(a, b - a).to_list()
        if chunk["fruit_id"].tolist() != expected:
            raise RuntimeError(f"chunk {k} holds other fruit than the plan")
        chunks.append(chunk)
        tables.append(pl.read_parquet(table_path))

    def stack(name):
        return np.concatenate([c[name] for c in chunks])

    mu = stack("mu")
    masks = stack("masks")
    np.save(out_dir / "latents.npy", mu)
    np.save(out_dir / "layer512_pooled.npy", stack("pooled"))
    np.save(out_dir / "masks.npy", masks)
    pool_grids(mu, masks, volumes["fruit_id"].to_list()).write_parquet(
        out_dir / "features.parquet"
    )
    # Chunks without decoded fruit have empty tables; keep only the others.
    non_empty = [t for t in tables if t.height > 0]
    if non_empty:
        recon = pl.concat(non_empty)
    else:
        recon = pl.DataFrame(schema={"fruit_id": pl.String})
    recon_row = {fid: i for i, fid in enumerate(recon["fruit_id"].to_list())}
    manifest = volumes.drop("path").with_columns(
        pl.col("fruit_id")
        .replace_strict(recon_row, default=None, return_dtype=pl.Int32)
        .alias("recon_row")
    )
    manifest.write_parquet(out_dir / "manifest.parquet")
    if recon.height:
        recon.write_parquet(out_dir / "reconstruction.parquet")
        np.save(out_dir / "planes.npy", stack("planes"))
        np.save(out_dir / "slice_mae_volume.npy", stack("slice_volume"))
        np.save(out_dir / "slice_mae_body.npy", stack("slice_body"))
    print(
        f"{out_dir.name}: assembled {volumes.height} fruit, {recon.height} reconstructed"
    )

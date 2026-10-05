import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="04 Encode latents")

with app.setup:
    import time

    import marimo as mo
    import nibabel as nib
    import numpy as np
    import polars as pl

    from pearlatent.config import load_config
    from pearlatent.device import describe_device
    from pearlatent.encoding import encode_volumes, load_encoder
    from pearlatent.pooling import body_mask, fruit_mask, pool_grids, scale_minmax
    from pearlatent.runs import new_run_dir, write_run_record
    from pearlatent.volumes import list_volumes


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 04 Encode latents

    **Question.** None: this notebook produces the latents (step 1 of the pipeline, CLAUDE.md
    section 5). It runs on the laptop as a script (CPU, about 5 seconds per fruit).

    **Inputs.** `ct_dir_2425`, `ct_dir_2526` (all volumes), `hugo_repo` (clone of Hugo's
    repository at commit 183112f), `encoder_dir` (Hugo's run folder with the checkpoint and
    `vae_hyperparameter.json`), from `config/local.toml`. Command-line arguments:
    `--encoder hugo_vae` (default) or `--encoder random_init` (randomly initialised encoder, the
    lower-bound reference of rule 7 of CHARTER.md), `--checkpoint` (default `checkpoint.pth`,
    what Hugo's `load_vae` loads), `--seed` (default 0, used by `random_init`), `--limit`
    (default 0 = all volumes), `--batch` (default `encode_batch_size` from the config),
    `--bf16 1` (bfloat16 on the GPU; off by default: full float32 precision, because browning
    is a small density shift and 1,100 fruit encode quickly anyway).

    **Outputs.** `derived/latents/<encoder>/<run_id>/`: `latents.npy` (fruit × 32³ float32, the
    mean `mu`), `layer512_pooled.npy` (fruit × 2 × 512 float32: mean and 99th percentile of each
    channel of the layer before the latent head, inside the fruit body mask), `masks.npy`,
    `manifest.parquet` (fruit_id, season, box, storage, file_stem, row), `features.parquet`
    (`mu` pooled with `pearlatent.pooling.pool_grids`), `run.json`. No labels are read.

    **Status.** draft. Tested on the laptop with `--encoder random_init --limit 2`
    (5 October 2026); not yet run with Hugo's weights.

    Run on the laptop:

    ```bash
    uv run python notebooks/04_encode_latents.py --encoder hugo_vae --limit 4    # first
    uv run python notebooks/04_encode_latents.py --encoder hugo_vae              # then all
    ```
    """)
    return


@app.cell
def _():
    cfg = load_config()
    _args = mo.cli_args()
    encoder_name = _args.get("encoder") or "hugo_vae"
    checkpoint = _args.get("checkpoint") or "checkpoint.pth"
    seed = int(_args.get("seed") or 0)
    limit = int(_args.get("limit") or 0)
    batch_size = int(_args.get("batch") or cfg.compute("encode_batch_size", 4))
    use_bf16 = (_args.get("bf16") or "0") == "1"
    if encoder_name not in ("hugo_vae", "random_init"):
        raise ValueError(f"--encoder must be hugo_vae or random_init, got {encoder_name!r}")
    device_info = describe_device(cfg.compute("device", "auto"))
    if use_bf16 and not device_info.bf16_supported:
        raise ValueError("--bf16 1 needs a GPU that supports bfloat16")
    mo.md(
        f"Encoder **{encoder_name}**, checkpoint `{checkpoint}`, seed {seed}, limit "
        f"{limit or 'none'}, batch {batch_size}, device **{device_info.name}**."
    )
    return (
        batch_size,
        cfg,
        checkpoint,
        device_info,
        encoder_name,
        limit,
        seed,
        use_bf16,
    )


@app.cell
def _(cfg, limit):
    volumes = pl.concat(
        [
            list_volumes(cfg.path("ct_dir_2425"), "2425"),
            list_volumes(cfg.path("ct_dir_2526"), "2526"),
        ]
    )
    if limit:
        volumes = volumes.head(limit)
    volumes = volumes.with_row_index("row")
    return (volumes,)


@app.cell
def _():
    run = mo.ui.run_button(label="Encode")
    run
    return (run,)


@app.cell
def _(
    batch_size,
    cfg,
    checkpoint,
    device_info,
    encoder_name,
    limit,
    run,
    seed,
    use_bf16,
    volumes,
):
    mo.stop(
        mo.running_in_notebook() and not run.value,
        mo.md("_Press the button to start. Encoding all fruit takes about 1.5 hours; run it as a script._"),
    )
    _encoder, _info = load_encoder(
        cfg.path("hugo_repo"),
        cfg.path("encoder_dir") if encoder_name == "hugo_vae" else None,
        device_info.device,
        checkpoint=checkpoint,
        random_init_seed=seed if encoder_name == "random_init" else None,
    )
    run_dir = new_run_dir(cfg.path("derived_dir") / "latents" / encoder_name, encoder_name)
    write_run_record(
        run_dir,
        {
            "step": "encode",
            "encoder": encoder_name,
            "limit": limit,
            "batch_size": batch_size,
            "device": device_info.name,
            "bf16": use_bf16,
            "mask": "body_mask(fruit_mask), threshold 0.1, min fraction 0.5, min 4 cells per slice",
            **_info,
        },
    )
    _n = volumes.height
    _mu = np.lib.format.open_memmap(
        run_dir / "latents.npy", mode="w+", dtype=np.float32, shape=(_n, 32, 32, 32)
    )
    _pooled = np.lib.format.open_memmap(
        run_dir / "layer512_pooled.npy",
        mode="w+",
        dtype=np.float32,
        shape=(_n, 2, 2 * _info["base_channel"]),
    )
    _masks = np.zeros((_n, 32, 32, 32), dtype=bool)
    _paths = volumes["path"].to_list()
    _start = time.perf_counter()
    for _i in mo.status.progress_bar(
        list(range(0, _n, batch_size)), title="Encoding", show_eta=True
    ):
        _j = min(_i + batch_size, _n)
        _scaled = np.stack(
            [scale_minmax(np.asarray(nib.load(p).dataobj)) for p in _paths[_i:_j]]
        )
        _masks[_i:_j] = [body_mask(fruit_mask(s)) for s in _scaled]
        _mu[_i:_j], _pooled[_i:_j] = encode_volumes(
            _encoder,
            _scaled,
            _masks[_i:_j],
            device_info.device,
            bf16=use_bf16,
        )
    _mu.flush()
    _pooled.flush()
    np.save(run_dir / "masks.npy", _masks)
    volumes.drop("path").write_parquet(run_dir / "manifest.parquet")
    pool_grids(np.asarray(_mu), _masks, volumes["fruit_id"].to_list()).write_parquet(
        run_dir / "features.parquet"
    )
    seconds_per_fruit = (time.perf_counter() - _start) / _n
    mo.md(f"Wrote `{run_dir}` ({_n} fruit, {seconds_per_fruit:.1f} s per fruit).")
    return (run_dir,)


@app.cell
def _(run_dir):
    _mu = np.load(run_dir / "latents.npy", mmap_mode="r")
    _pooled = np.load(run_dir / "layer512_pooled.npy", mmap_mode="r")
    _feat = pl.read_parquet(run_dir / "features.parquet")
    mo.vstack(
        [
            mo.md(
                f"`latents.npy` {_mu.shape}, finite: {bool(np.isfinite(_mu).all())}; "
                f"`layer512_pooled.npy` {_pooled.shape}, finite: "
                f"{bool(np.isfinite(_pooled).all())}; `mu` range "
                f"{float(np.min(_mu)):.3g} to {float(np.max(_mu)):.3g}."
            ),
            mo.ui.table(_feat, selection=None, label="Pooled mu"),
        ]
    )
    return


if __name__ == "__main__":
    app.run()

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="04 Encode latents")

with app.setup:
    import marimo as mo
    import numpy as np
    import polars as pl

    from pearlatent.config import load_config
    from pearlatent.device import describe_device
    import json
    import time

    from pearlatent.encode_run import (
        CHUNK_SIZE,
        assemble,
        encode_in_chunks,
        keep_awake,
    )
    from pearlatent.encoding import load_vae
    from pearlatent.hugo_split import TRAINING_SAMPLE_SIZE, hugo_training_sample
    from pearlatent.reconstruction import CORE_RADIUS_FRACTION
    from pearlatent.runs import git_state, new_run_dir, write_run_record
    from pearlatent.sealed import read_sealed_fruit
    from pearlatent.volumes import list_volumes


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 04 Encode latents

    **Question.** None: this notebook produces the latents (step 1 of the pipeline, CLAUDE.md
    section 5) and, in the same pass, the reconstruction numbers of Step 0 (`x_hat =
    vae.decode(mu)`, never `vae(x)`, which samples z). It runs on the laptop as a script (CPU,
    about 4 seconds per fruit to encode, about 56 seconds more to decode; measured 13 October
    2026). Because decoding is slow, only the fruit needed now are decoded (Ruben, 13 October
    2026): the open 2526 fruit, Hugo's 66 validation fruit and a random sample of 66 of his 528
    training fruit. All other fruit are encoded only; the sealed fruit are decoded at the final
    test. The work is saved in chunks of 20 fruit, so the run can be stopped and resumed.

    **Inputs.** `ct_dir_2425`, `ct_dir_2526` (all volumes), `hugo_repo` (clone of Hugo's
    repository at commit 183112f), `encoder_dir` (Hugo's run folder with the checkpoint and
    `vae_hyperparameter.json`), from `config/local.toml`; `derived/sealed_fruit.csv` and
    `derived/hugo_vae_split_2425/hugo_vae_split_2425.csv`. Command-line arguments:
    `--encoder hugo_vae` (default) or `--encoder random_init` (randomly initialised encoder, the
    lower-bound reference of rule 7 of CHARTER.md), `--checkpoint` (default `best.pth`,
    the only checkpoint Hugo sent, 13 October 2026), `--seed` (default 0, used by `random_init`), `--limit`
    (default 0 = all volumes), `--batch` (default `encode_batch_size` from the config),
    `--bf16 1` (bfloat16 on the GPU; off by default: full float32 precision, because browning
    is a small density shift), `--resume <run name>` (continue that run, for example
    `20261013T200000Z_hugo_vae`), `--stop-after-hours H` (stop cleanly between chunks after H
    hours; default: run to the end).

    **Outputs.** `derived/latents/<encoder>/<run_id>/` for the open fruit and
    `derived/sealed/latents/<encoder>/<run_id>/` for the sealed fruit (2526 test boxes and Hugo's
    2425 test fruit; no notebook reads that folder). Each holds `latents.npy` (fruit × 32³
    float32, the mean `mu`), `layer512_pooled.npy` (fruit × 2 × 512 float32: mean and 99th
    percentile of each channel of the layer before the latent head, inside the fruit body mask),
    `masks.npy`, `manifest.parquet` (fruit_id, season, box, storage, file_stem, row,
    hugo_training, hugo_split, sealed, decode, recon_row), `features.parquet` (`mu` pooled with
    `pearlatent.pooling.pool_grids`), `run.json`, `sessions.jsonl` (one line per night); for the
    decoded fruit also `reconstruction.parquet`, `slice_mae_volume.npy`, `slice_mae_body.npy` and
    `planes.npy`. While a run is unfinished, the folder holds `plan.parquet` and `parts/` (see
    `pearlatent.encode_run`). The regions (body, core
    with radius one third of the widest body radius) are defined in `pearlatent.reconstruction`,
    fixed on 13 October 2026 before any reconstruction was seen. No labels are read.

    **Status.** draft. Tested on the laptop with `--encoder random_init --limit 2`
    (5 October 2026); reconstruction added 13 October 2026.

    Run on the laptop (plugged in):

    ```bash
    uv run python notebooks/04_encode_latents.py --encoder hugo_vae --stop-after-hours 5   # night 1
    uv run python notebooks/04_encode_latents.py --encoder hugo_vae --resume <run name>    # night 2
    uv run python notebooks/04_encode_latents.py --encoder random_init                     # encode only
    ```
    """)
    return


@app.cell
def _():
    cfg = load_config()
    _args = mo.cli_args()
    encoder_name = _args.get("encoder") or "hugo_vae"
    checkpoint = _args.get("checkpoint") or "best.pth"
    seed = int(_args.get("seed") or 0)
    limit = int(_args.get("limit") or 0)
    batch_size = int(_args.get("batch") or cfg.compute("encode_batch_size", 4))
    use_bf16 = (_args.get("bf16") or "0") == "1"
    resume = _args.get("resume") or ""
    stop_after_hours = float(_args.get("stop-after-hours") or 0)
    if encoder_name not in ("hugo_vae", "random_init"):
        raise ValueError(f"--encoder must be hugo_vae or random_init, got {encoder_name!r}")
    device_info = describe_device(cfg.compute("device", "auto"))
    if use_bf16 and not device_info.bf16_supported:
        raise ValueError("--bf16 1 needs a GPU that supports bfloat16")
    mo.md(
        f"Encoder **{encoder_name}**, checkpoint `{checkpoint}`, seed {seed}, limit "
        f"{limit or 'none'}, batch {batch_size}, device **{device_info.name}**, resume "
        f"`{resume or 'no'}`, stop after {stop_after_hours or 'no limit'} hours."
    )
    return (
        batch_size,
        cfg,
        checkpoint,
        device_info,
        encoder_name,
        limit,
        resume,
        seed,
        stop_after_hours,
        use_bf16,
    )


@app.cell
def _(cfg, encoder_name, limit):
    volumes = pl.concat(
        [
            list_volumes(cfg.path("ct_dir_2425"), "2425"),
            list_volumes(cfg.path("ct_dir_2526"), "2526"),
        ]
    )
    if limit:
        volumes = volumes.head(limit)
    _sealed = read_sealed_fruit(cfg.path("derived_dir") / "sealed_fruit.csv")
    _split = pl.read_csv(
        cfg.path("derived_dir") / "hugo_vae_split_2425" / "hugo_vae_split_2425.csv"
    ).select("fruit_id", "hugo_split")
    # The training fruit decoded for the Step 0 comparison (seed 369, see hugo_split.py).
    _sample = hugo_training_sample(_split)
    volumes = (
        volumes.with_row_index("row")
        .join(_split, on="fruit_id", how="left", validate="1:1")
        .with_columns(
            (pl.col("hugo_split") == "train").fill_null(False).alias("hugo_training"),
            pl.col("fruit_id").is_in(_sealed["fruit_id"].implode()).alias("sealed"),
        )
    )
    # Decode only what Step 0 needs now (Ruben, 13 October 2026); the random encoder has no
    # trained decoder, so nothing is decoded for it.
    _needed_now = (
        (pl.col("season") == "2526")
        | (pl.col("hugo_split") == "validation").fill_null(False)
        | pl.col("fruit_id").is_in(_sample)
    )
    volumes = volumes.with_columns(
        (_needed_now & ~pl.col("sealed") & (encoder_name == "hugo_vae")).alias("decode")
    )
    # Every volume is encoded; sealed fruit go to their own folder (charter rule 3).
    volumes_open = volumes.filter(~pl.col("sealed"))
    volumes_sealed = volumes.filter(pl.col("sealed"))
    print(
        f"volumes: {volumes.height} in; open {volumes_open.height}, sealed "
        f"{volumes_sealed.height}; Hugo VAE training fruit "
        f"{int(volumes['hugo_training'].sum())}, of which in the decoded sample "
        f"{int((volumes['hugo_training'] & volumes['decode']).sum())} "
        f"(sample size {TRAINING_SAMPLE_SIZE}); to decode {int(volumes['decode'].sum())}"
    )
    return volumes_open, volumes_sealed


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
    resume,
    run,
    seed,
    stop_after_hours,
    use_bf16,
    volumes_open,
    volumes_sealed,
):
    mo.stop(
        mo.running_in_notebook() and not run.value,
        mo.md("_Press the button to start. The full run takes hours; run it as a script._"),
    )
    _vae, _info = load_vae(
        cfg.path("hugo_repo"),
        cfg.path("encoder_dir") if encoder_name == "hugo_vae" else None,
        device_info.device,
        checkpoint=checkpoint,
        random_init_seed=seed if encoder_name == "random_init" else None,
    )
    _latents_base = cfg.path("derived_dir") / "latents" / encoder_name
    if resume:
        run_dir = _latents_base / resume
        if not (run_dir / "run.json").exists():
            raise FileNotFoundError(f"no run to resume at {run_dir}")
    else:
        run_dir = new_run_dir(_latents_base, encoder_name)
    _sealed_dir = cfg.path("derived_dir") / "sealed" / "latents" / encoder_name / run_dir.name
    _sealed_dir.mkdir(parents=True, exist_ok=True)
    _params = {
        "step": "encode",
        "encoder": encoder_name,
        "limit": limit,
        "batch_size": batch_size,
        "device": device_info.name,
        "bf16": use_bf16,
        "mask": "body_mask(fruit_mask), threshold 0.1, min fraction 0.5, min 4 cells per slice",
        "reconstruction": "x_hat = vae.decode(mu); whole volume, body voxels, core",
        "core_radius_fraction": CORE_RADIUS_FRACTION,
        "decoded": "open 2526 fruit, Hugo validation fruit, sample of his training fruit",
        "training_sample_size": TRAINING_SAMPLE_SIZE,
        "training_sample_seed": 369,
        "chunk_size": CHUNK_SIZE,
        **_info,
    }
    # These settings must be the same on every night of one run.
    _must_match = ("encoder", "limit", "core_radius_fraction", "chunk_size", "checkpoint",
                   "checkpoint_epoch", "hugo_commit", "training_sample_size")
    _groups = (("open", volumes_open, run_dir), ("sealed", volumes_sealed, _sealed_dir))
    for _name, _group, _dir in _groups:
        if resume:
            _first = json.loads((_dir / "run.json").read_text())["params"]
            _differ = [k for k in _must_match if _first.get(k) != _params.get(k)]
            if _differ:
                raise RuntimeError(f"cannot resume {_dir.name}: settings differ: {_differ}")
        else:
            write_run_record(_dir, {**_params, "group": _name, "n_fruit": _group.height})
    _deadline = time.time() + 3600 * stop_after_hours if stop_after_hours else None
    _started = time.strftime("%Y-%m-%d %H:%M:%S")
    _finished = {}
    with keep_awake():
        for _name, _group, _dir in _groups:
            _finished[_name] = encode_in_chunks(
                _vae,
                _group,
                _dir,
                device_info.device,
                batch_size,
                bf16=use_bf16,
                deadline=_deadline,
            )
            if _finished[_name]:
                assemble(_group, _dir)
            _session = {
                "started": _started,
                "ended": time.strftime("%Y-%m-%d %H:%M:%S"),
                "git": git_state(),
                "stop_after_hours": stop_after_hours,
                "finished": _finished[_name],
            }
            with open(_dir / "sessions.jsonl", "a", encoding="utf-8") as _log:
                _log.write(json.dumps(_session) + "\n")
    complete = all(_finished.get(_n, False) for _n, _, _ in _groups)
    print(
        f"run {run_dir.name}: "
        + ("complete" if complete else f"not finished; resume with --resume {run_dir.name}")
    )
    mo.md(f"Run `{run_dir.name}`: {'complete' if complete else 'not finished'}.")
    return complete, run_dir


@app.cell
def _(complete, run_dir):
    mo.stop(not complete, mo.md("_The run is not finished yet; nothing to check._"))
    _recon_file = run_dir / "reconstruction.parquet"
    _recon_text = "No reconstruction (random encoder)."
    if _recon_file.exists():
        _recon = pl.read_parquet(_recon_file)
        _numbers = _recon.select(pl.exclude("fruit_id")).to_numpy()
        _recon_text = (
            f"`reconstruction.parquet`: {_recon.height} rows, all finite: "
            f"{bool(np.isfinite(_numbers).all())}."
        )
    _mu = np.load(run_dir / "latents.npy", mmap_mode="r")
    _pooled = np.load(run_dir / "layer512_pooled.npy", mmap_mode="r")
    _feat = pl.read_parquet(run_dir / "features.parquet")
    mo.vstack(
        [
            mo.md(
                f"`latents.npy` {_mu.shape}, finite: {bool(np.isfinite(_mu).all())}; "
                f"`layer512_pooled.npy` {_pooled.shape}, finite: "
                f"{bool(np.isfinite(_pooled).all())}; `mu` range "
                f"{float(np.min(_mu)):.3g} to {float(np.max(_mu)):.3g}. {_recon_text}"
            ),
            mo.ui.table(_feat, selection=None, label="Pooled mu"),
        ]
    )
    return


if __name__ == "__main__":
    app.run()

"""Decode counterfactual steps of one fruit and re-encode them (notebook 17; Ruben, 9 November 2026).

    uv run python -m pearlatent.counterfactual_run --fruit 2526_C23 --direction cavity_free \
        --t 0.5 0.0 --plain derived/step0_examples/20261109T174632Z_examples

For every t: move the fruit's saved `mu` along the direction b to the score of t
(`pearlatent.counterfactual`), decode with `vae.decode` (about 5 minutes and 18 GB on the Mac CPU,
9 November 2026), min-max scale the decoded volume as every CT is scaled before encoding, encode
it again and let the same model predict the new `mu`. The re-encoded prediction shows whether the
decoded image itself carries the change, not only the moved latent.

Control: the plain reconstruction (no shift, from an earlier decode, saved as float16) is
re-encoded the same way. Its prediction shows how much the decode-encode round trip alone moves
the score.

Writes `derived/counterfactuals/<timestamp>_<fruit>_<direction>/`: one `step_t<t>.npz` per step as
soon as it is decoded (so a stop loses at most the current step), `steps.parquet` and `run.json`.
Open fruit only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from pearlatent.counterfactual import GradeScale, plsda_direction, shift_to_score
from pearlatent.runs import ENCODING_RUN, ENCODING_RUN_DIR

# LVs chosen by Ruben for the two full-grid latent models (notebooks 08 and 14).
DIRECTIONS = {
    "cavity_free": {
        "target": "browning_2plus_cavity_free",
        "n_lv": 5,
        "grade": "browning",
    },
    "general": {"target": None, "n_lv": 4, "grade": "defective"},
}


def fit_direction(cfg, name: str):
    """The direction and its t scale, fitted on all open fruit of its target (12 boxes).

    Out: (Direction, GradeScale, the fruit table it was fitted on, the latent rows).
    """
    from pearlatent.feature_sets import load_features
    from pearlatent.fruit_table import drop_faulty_volumes, open_fruit
    from pearlatent.targets import target_table

    spec = DIRECTIONS[name]
    kept = drop_faulty_volumes(
        open_fruit(cfg).filter(pl.col("season") == "2526"),
        cfg.path("derived_dir") / "volume_decisions.csv",
    )
    if spec["target"] is None:
        fruit = kept.with_columns(y=(pl.col("defective") >= 2).cast(pl.Int8))
    else:
        fruit = target_table(kept, spec["target"])
    x = load_features(cfg, "hugo_vae", fruit["fruit_id"].to_list()).astype(float)
    direction = plsda_direction(x, fruit["y"].to_numpy(), spec["n_lv"])
    fitted = direction.predict(x)
    grade = fruit[spec["grade"]].to_numpy()
    scale = GradeScale(
        float(fitted[grade == 0].mean()), float(fitted[grade == 3].mean())
    )
    return direction, scale, fruit, x


def reencode(vae, volume: np.ndarray, device: str) -> np.ndarray:
    """`mu` (32³) of a decoded volume, after min-max scaling as Hugo's loader does."""
    import torch

    from pearlatent.pooling import scale_minmax

    scaled = scale_minmax(volume)
    with torch.inference_mode():
        mu, _ = vae.encoder(torch.from_numpy(scaled[None, None]).to(device))
    return mu[0, 0].float().cpu().numpy()


def main() -> None:
    import torch

    from pearlatent.config import load_config
    from pearlatent.device import get_device
    from pearlatent.encoding import load_vae
    from pearlatent.runs import new_run_dir, write_run_record

    parser = argparse.ArgumentParser()
    parser.add_argument("--fruit", required=True)
    parser.add_argument("--direction", choices=list(DIRECTIONS), required=True)
    parser.add_argument("--t", type=float, nargs="+", required=True)
    parser.add_argument("--plain", type=Path, help="step0_examples run with this fruit")
    args = parser.parse_args()

    cfg = load_config()
    direction, scale, _, _ = fit_direction(cfg, args.direction)
    manifest = pl.read_parquet(
        cfg.path("derived_dir") / ENCODING_RUN_DIR / "manifest.parquet"
    ).with_row_index("position")
    match = manifest.filter(pl.col("fruit_id") == args.fruit)
    if match.height != 1:
        raise ValueError(f"{args.fruit} is not an open fruit of {ENCODING_RUN}")
    latents = np.load(
        cfg.path("derived_dir") / ENCODING_RUN_DIR / "latents.npy", mmap_mode="r"
    )
    mu = np.array(latents[match["position"][0]], dtype=np.float64)
    start = float(direction.predict(mu.ravel()))
    print(
        f"{args.fruit}: score {start:.3f}, t {scale.t(start):.2f}"
        f" (grade 0 mean {scale.grade0_mean:.3f}, grade 3 mean {scale.grade3_mean:.3f})"
    )

    device = get_device(cfg.compute("device", "cpu"))
    vae, info = load_vae(cfg.path("hugo_repo"), cfg.path("encoder_dir"), device)
    out_dir = new_run_dir(
        cfg.path("derived_dir") / "counterfactuals", f"{args.fruit}_{args.direction}"
    )
    write_run_record(
        out_dir,
        {
            "fruit": args.fruit,
            "direction": args.direction,
            "n_lv": direction.n_components,
            "t_steps": args.t,
            "grade0_mean_score": scale.grade0_mean,
            "grade3_mean_score": scale.grade3_mean,
            "start_score": start,
            "plain_reconstruction": str(args.plain) if args.plain else None,
            "encoding_run": ENCODING_RUN,
            **info,
        },
        root=cfg.root,
    )
    rows = []
    if args.plain is not None:
        with np.load(cfg.root / args.plain / "volumes.npz") as saved:
            k = list(saved["fruit_id"]).index(args.fruit)
            plain = saved["reconstruction"][k].astype(np.float32)
        mu_back = reencode(vae, plain, device)
        rows.append(
            {
                "step": "plain",
                "t_target": None,
                "score_target": None,
                "score_reencoded": float(direction.predict(mu_back.ravel())),
                "change_norm": 0.0,
            }
        )
        print(
            f"plain reconstruction re-encoded: score {rows[-1]['score_reencoded']:.3f}"
        )
    for t in args.t:
        target = scale.score(t)
        moved = shift_to_score(mu.ravel(), direction, target).reshape(mu.shape)
        print(f"t {t}: decoding ...", flush=True)
        with torch.inference_mode():
            volume = (
                vae.decode(
                    torch.from_numpy(moved.astype(np.float32))[None, None].to(device)
                )
                .float()
                .cpu()
                .numpy()[0, 0]
            )
        mu_back = reencode(vae, volume, device)
        rows.append(
            {
                "step": f"t={t}",
                "t_target": t,
                "score_target": target,
                "score_reencoded": float(direction.predict(mu_back.ravel())),
                "change_norm": float(np.linalg.norm(moved - mu)),
            }
        )
        np.savez_compressed(
            out_dir / f"step_t{t}.npz",
            reconstruction=volume.astype(np.float16),
            moved_mu=moved.astype(np.float32),
            reencoded_mu=mu_back,
        )
        pl.DataFrame(rows).write_parquet(out_dir / "steps.parquet")
        print(
            f"t {t}: target score {target:.3f}, re-encoded score {rows[-1]['score_reencoded']:.3f}"
        )
    pl.DataFrame(rows).write_parquet(out_dir / "steps.parquet")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()

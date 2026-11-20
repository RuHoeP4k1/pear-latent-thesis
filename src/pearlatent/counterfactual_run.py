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
from pearlatent.plsda import CUTOFF
from pearlatent.runs import ENCODING_RUN, ENCODING_RUN_DIR

# LVs chosen by Ruben for the two full-grid latent models (notebooks 08 and 14).
DIRECTIONS = {
    "cavity_free": {
        "target": "browning_2plus_cavity_free",
        "n_lv": 5,
        "grade": "browning",
    },
    "general": {"target": None, "n_lv": 4, "grade": "defective"},
    # EPO versions (20 November 2026, `pearlatent.epo`): the first principal component of the
    # healthy fruit (the size axis) is removed before the fit, so b is orthogonal to it. The
    # number of LVs is chosen by Ruben on the EPO curve (notebook 22) and passed as --n-lv.
    "cavity_free_epo": {
        "target": "browning_2plus_cavity_free",
        "n_lv": None,
        "grade": "browning",
        "epo": True,
    },
    "general_epo": {"target": None, "n_lv": None, "grade": "defective", "epo": True},
}


def fit_direction(cfg, name: str, n_lv: int | None = None):
    """The direction and its t scale, fitted on all open fruit of its target (12 boxes).

    `n_lv` overrides the number of LVs of the direction; the EPO directions require it.

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
    n_components = n_lv if n_lv is not None else spec["n_lv"]
    if n_components is None:
        raise ValueError(f"direction {name} needs --n-lv (chosen in notebook 22)")
    # Healthy = defective grade 0 (for the cavity-free target: browning 0 and cavity 0).
    healthy = (fruit["defective"] == 0).to_numpy() if spec.get("epo") else None
    direction = plsda_direction(
        x, fruit["y"].to_numpy(), n_components, epo_healthy=healthy
    )
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
    parser.add_argument("--t", type=float, nargs="*", default=[])
    # Ruben, 20 November 2026: mirror the fruit across the decision boundary (score = CUTOFF):
    # target score = 2 × CUTOFF − start score, the same distance on the other side.
    parser.add_argument("--flip", action="store_true", help="mirror across score 0.5")
    parser.add_argument("--plain", type=Path, help="step0_examples run with this fruit")
    parser.add_argument("--n-lv", type=int, help="number of LVs (required for *_epo)")
    args = parser.parse_args()
    if not args.t and not args.flip:
        parser.error("give --t values, --flip, or both")

    cfg = load_config()
    direction, scale, _, _ = fit_direction(cfg, args.direction, args.n_lv)
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

    t_steps = list(args.t)
    if args.flip:
        flip_target = 2 * CUTOFF - start
        # Rounded to 3 decimals so the step file name stays readable (step_t-0.247.npz).
        t_steps.append(round(scale.t(flip_target), 3))
        print(
            f"flip across {CUTOFF}: target score {flip_target:.3f}, t {t_steps[-1]:.2f}"
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
            "t_steps": t_steps,
            "flip": args.flip,
            "grade0_mean_score": scale.grade0_mean,
            "grade3_mean_score": scale.grade3_mean,
            "start_score": start,
            "plain_reconstruction": str(args.plain) if args.plain else None,
            "epo": direction.removed is not None,
            "encoding_run": ENCODING_RUN,
            **info,
        },
        root=cfg.root,
    )
    if direction.removed is not None:
        np.save(out_dir / "epo_directions.npy", direction.removed)
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
    for t in t_steps:
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
                # Check for EPO directions: the step's part along the removed size axis
                # (0 by construction) and the re-encoded fruit's change along it.
                "size_axis_step": float((moved - mu).ravel() @ direction.removed[:, 0])
                if direction.removed is not None
                else None,
                "size_axis_reencoded_change": float(
                    (mu_back - mu).ravel() @ direction.removed[:, 0]
                )
                if direction.removed is not None
                else None,
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

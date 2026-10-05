"""Encoding CT volumes with Hugo's frozen encoder (step 1 of the pipeline, CLAUDE.md section 5).

Hugo's code is not copied into this repository (no licence file). It is imported from a separate
clone of github.com/Hugo-Li-Junyan/Synthetic_CT_pear at the pinned commit, whose path is
`hugo_repo` in config/local.toml. `import_hugo_vae` refuses a clone at another commit.

Architecture (component/vae.py at commit 183112f, input 128³, featuremap_size 32,
base_channel 256): `encoder.feature_layers` = stride-2 convolution 1 -> 256 channels
(128³ -> 64³), stride-2 convolution 256 -> 512 (64³ -> 32³) with instance norm, then one residual
block; `encoder.mu_head` = 3³ convolution 512 -> 1. We keep `mu` in full (32³ per fruit) and the
512-channel layer only pooled inside the fruit mask (mean and 99th percentile per channel), since
in full it is 64 MB per fruit.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

PINNED_COMMIT = "183112f"
INPUT_SHAPE = (1, 128, 128, 128)
# Defaults of the trained model (CLAUDE.md section 3); a run folder's
# vae_hyperparameter.json overrides them.
DEFAULT_FEATUREMAP_SIZE = 32
DEFAULT_BASE_CHANNEL = 256


def hugo_commit(repo: str | Path) -> str:
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


def import_hugo_vae(repo: str | Path):
    """Return Hugo's `VAE` class from a clone at the pinned commit."""
    repo = Path(repo)
    commit = hugo_commit(repo)
    if not commit.startswith(PINNED_COMMIT):
        raise RuntimeError(
            f"{repo} is at commit {commit[:7]}, expected {PINNED_COMMIT}. "
            f"Run `git -C {repo} checkout {PINNED_COMMIT}`."
        )
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from component.vae import VAE

    return VAE


def load_encoder(
    repo: str | Path,
    model_dir: str | Path | None,
    device: str,
    checkpoint: str = "checkpoint.pth",
    random_init_seed: int | None = None,
):
    """Hugo's encoder in evaluation mode, and a dictionary describing what was loaded.

    model_dir: run folder with `vae_hyperparameter.json` and the checkpoint, as Hugo's
    `utils/load_models.py` `load_vae` expects. None gives a randomly initialised encoder with
    `random_init_seed` (the lower-bound reference point, rule 7 of CHARTER.md).
    """
    import torch

    VAE = import_hugo_vae(repo)
    info = {
        "hugo_commit": hugo_commit(repo),
        "checkpoint": None,
        "random_init_seed": None,
    }
    featuremap_size, base_channel = DEFAULT_FEATUREMAP_SIZE, DEFAULT_BASE_CHANNEL
    if model_dir is not None:
        hp = json.loads((Path(model_dir) / "vae_hyperparameter.json").read_text())
        featuremap_size = hp["vae_featuremap_size"]
        base_channel = hp["vae_base_channel"]
        info["hyperparameters"] = hp
    elif random_init_seed is None:
        raise ValueError("give model_dir, or random_init_seed for a random encoder")
    if model_dir is None:
        torch.manual_seed(random_init_seed)
        info["random_init_seed"] = random_init_seed
    vae = VAE(
        input_shape=INPUT_SHAPE,
        featuremap_size=featuremap_size,
        base_channel=base_channel,
        flatten_latent_dim=None,
        with_residual=True,
    )
    if model_dir is not None:
        path = Path(model_dir) / checkpoint
        state = torch.load(path, map_location="cpu")
        vae.load_state_dict(state["vae_state_dict"])
        info["checkpoint"] = str(path)
        info["checkpoint_epoch"] = state.get("epoch")
        info["checkpoint_random_state"] = state.get("random_state")
    encoder = vae.encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    info |= {"featuremap_size": featuremap_size, "base_channel": base_channel}
    return encoder, info


def encode_volumes(
    encoder, scaled: np.ndarray, masks: np.ndarray, device: str, bf16=False
):
    """Encode a batch.

    scaled: (B, 128, 128, 128) float32 volumes, min-max scaled to [0, 1].
    masks: (B, 32, 32, 32) boolean fruit masks (`pearlatent.pooling.fruit_mask`).
    Returns `mu` (B, 32, 32, 32) and the 512-channel layer pooled inside the mask,
    (B, 2, C): index 0 the mean per channel, index 1 the 99th percentile per channel.
    """
    import torch

    x = torch.from_numpy(np.ascontiguousarray(scaled, dtype=np.float32)).unsqueeze(1)
    m = torch.from_numpy(np.asarray(masks, dtype=bool)).to(device)
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bf16):
            features = encoder.feature_layers(x.to(device))
            mu = encoder.mu_head(features)
        features = features.float()
        pooled = []
        for b in range(features.shape[0]):
            inside = features[b][:, m[b]]  # (C, n_cells)
            pooled.append(
                torch.stack([inside.mean(dim=1), torch.quantile(inside, 0.99, dim=1)])
            )
        return (
            mu.float()[:, 0].cpu().numpy(),
            torch.stack(pooled).cpu().numpy(),
        )

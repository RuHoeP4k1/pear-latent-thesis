"""Per-fruit reconstruction error of Hugo's VAE (Step 0 of CHARTER.md; handoff of 13 October, B5).

The reconstruction is `x_hat = vae.decode(mu)`, never `vae(x)`: Hugo's forward pass samples
z = mu + noise, so its output changes from run to run.

Regions, fixed on 13 October 2026 before any reconstruction was looked at:

- whole volume: all 128³ voxels, including the zero background (comparable with the preprint, but
  the background is easy to reconstruct and makes the numbers look better than the fruit is);
- body: fruit voxels (scaled intensity > `MASK_THRESHOLD`) inside the 32³ body mask of
  `pearlatent.pooling.body_mask` (largest connected part, stalk slices removed), with enclosed
  holes filled, so internal cavities count as body. The same body as the latent features use;
- core: body voxels within `CORE_RADIUS_FRACTION` of the body's widest radius from the
  stem-calyx axis. The axis is the line along the third array axis through the mean x and mean y
  of the body voxels (preprocessing put the stem-calyx axis along the third axis, notebook 02).
  The widest radius is sqrt(largest slice area / pi). One third: internal browning develops
  around the core and seed cavity, the inner part of the fruit.
"""

from __future__ import annotations

import numpy as np

from pearlatent.pooling import FACTOR, MASK_THRESHOLD

# Fraction of the body's widest radius that defines the core (fixed before any result).
CORE_RADIUS_FRACTION = 1 / 3


def body_voxel_mask(scaled: np.ndarray, body_cells: np.ndarray) -> np.ndarray:
    """Voxel mask (128³) of the fruit body.

    In: the scaled volume and its 32³ body mask. Out: fruit voxels inside the body cells, with
    enclosed holes (cavities) filled.
    """
    from scipy import ndimage

    # Each 32³ cell covers FACTOR³ voxels; repeat it along every axis.
    cells_as_voxels = body_cells.repeat(FACTOR, 0).repeat(FACTOR, 1).repeat(FACTOR, 2)
    fruit = np.asarray(scaled) > MASK_THRESHOLD
    return ndimage.binary_fill_holes(fruit & cells_as_voxels)


def core_voxel_mask(body: np.ndarray) -> tuple[np.ndarray, dict]:
    """Voxel mask of the core, and the numbers that define it (axis and radius, in voxels)."""
    xs, ys, _ = np.nonzero(body)
    axis_x, axis_y = float(xs.mean()), float(ys.mean())
    largest_slice_area = int(body.sum(axis=(0, 1)).max())
    widest_radius = float(np.sqrt(largest_slice_area / np.pi))
    core_radius = CORE_RADIUS_FRACTION * widest_radius
    gx, gy = np.meshgrid(
        np.arange(body.shape[0]), np.arange(body.shape[1]), indexing="ij"
    )
    within = (gx - axis_x) ** 2 + (gy - axis_y) ** 2 <= core_radius**2
    core = body & within[:, :, None]
    geometry = {
        "axis_x": axis_x,
        "axis_y": axis_y,
        "widest_radius_voxels": widest_radius,
        "core_radius_voxels": core_radius,
    }
    return core, geometry


def masked_mae(abs_residual: np.ndarray, mask: np.ndarray) -> float:
    """Mean absolute residual inside `mask`; NaN when the mask is empty."""
    n = int(mask.sum())
    return float(abs_residual[mask].mean()) if n else float("nan")


def slice_mae(abs_residual: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Mean absolute residual per slice along the third axis (128 values).

    With `mask`, only voxels inside it count, and slices without such voxels give NaN.
    """
    if mask is None:
        return abs_residual.mean(axis=(0, 1))
    totals = (abs_residual * mask).sum(axis=(0, 1))
    counts = mask.sum(axis=(0, 1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 0, totals / np.maximum(counts, 1), np.nan)


def central_planes(volume: np.ndarray, centre: tuple[int, int, int]) -> np.ndarray:
    """The three planes through `centre`: (3, 128, 128) for axes 0, 1 and 2 held fixed."""
    i, j, k = centre
    return np.stack([volume[i, :, :], volume[:, j, :], volume[:, :, k]])


def body_centre(body: np.ndarray) -> tuple[int, int, int]:
    """Rounded mean position of the body voxels, used for the central planes."""
    return tuple(int(round(float(c.mean()))) for c in np.nonzero(body))


def whole_volume_metrics(x, x_hat) -> dict:
    """MAE, SSIM and PSNR over the whole volume with Hugo's `utils/metrics.py` (imported).

    In: torch tensors (1, 1, 128, 128, 128) with values in [0, 1]. Hugo's evaluation averages
    PSNR per batch of 4; here it is per fruit, so a mean over fruit can differ slightly.
    """
    # Hugo's clone, put on sys.path by `pearlatent.encoding.import_hugo_vae`.
    from utils.metrics import mae, psnr, ssim

    return {
        "mae_volume": float(mae(x, x_hat)),
        "ssim_volume": float(ssim(x, x_hat, data_range=1.0)),
        "psnr_volume": float(psnr(x, x_hat, data_range=1.0)),
    }

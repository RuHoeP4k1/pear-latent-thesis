"""The 10 tissue features of Van De Looverbosch et al. (2020), on the 128³ volumes (2 November 2026).

Source: Van De Looverbosch, T., et al. (2020). Nondestructive internal quality inspection of pear
fruit by X-ray CT using machine learning. *Food Control* 113, 107170, section 2.3.1. Their
features are global intensity statistics over regions of the fruit; we use them as a comparison
row for the latent (Ruben, 2 November 2026), because InstanceNorm passes such whole-fruit
statistics into every latent cell (notebook 09).

Masks, in the order of the paper:

1. tissue: voxels above a global Otsu threshold of the whole volume (Otsu 1979);
2. fruit: the tissue mask with enclosed holes filled;
3. internal air: fruit minus tissue (cavities and the air spaces of the core);
4. low- and high-density tissue: a tissue voxel is low density when it is below the local mean
   intensity of the tissue in a 31 × 31 × 31 voxel window, otherwise high density. Pear tissue is
   denser around the core and under the skin than in between; comparing each voxel with its own
   neighbourhood separates locally darker tissue from that normal gradient, which one global
   threshold cannot do.

Features: internal air volume over fruit volume; mean and standard deviation of the intensity in
the fruit, tissue, low-density and high-density masks (8 numbers); the two-sample
Kolmogorov-Smirnov statistic between the intensities of low- and high-density tissue (the paper's
measure of how uneven the tissue is).

Differences from the paper, all chosen on 2 November 2026:

- Only the largest connected part of the tissue is kept (stray objects, foam), as in
  `pearlatent.pooling.body_mask`. The paper does not mention this step.
- The stalk stays in the masks (Hugo, 13 October 2026: no stalk removal).
- The window stays 31 voxels. The paper's voxels are 0.515 mm (about 16 mm window); ours are
  the fruit length over 128 (about 0.6 to 0.7 mm, different per fruit), so about 20 mm.
- Intensities are the min-max scaled volume (as the encoder sees it), not raw values.
"""

from __future__ import annotations

import numpy as np

from pearlatent.runs import ENCODING_RUN, ENCODING_RUN_DIR

WINDOW = 31  # voxels per side of the local-mean window (paper, section 2.3.1)
FEATURE_NAMES = [
    "air_fraction",
    "fruit_mean",
    "fruit_sd",
    "tissue_mean",
    "tissue_sd",
    "low_density_mean",
    "low_density_sd",
    "high_density_mean",
    "high_density_sd",
    "ks_low_high",
]


def tissue_masks(scaled: np.ndarray, window: int = WINDOW) -> dict[str, np.ndarray]:
    """The five boolean masks of the paper, each the shape of `scaled`.

    In: a volume min-max scaled to [0, 1]. Out: a dict with `tissue`, `fruit`, `air`,
    `low_density`, `high_density` (the last two split `tissue`) and the Otsu threshold.
    """
    from scipy import ndimage
    from skimage.filters import threshold_otsu

    volume = np.asarray(scaled, dtype=np.float32)
    # Otsu: the threshold that maximises the variance between the two intensity classes,
    # computed on the histogram of all voxels (air and fruit), as the paper's "3D global" Otsu.
    threshold = float(threshold_otsu(volume))
    above = volume > threshold
    labels, n = ndimage.label(above)
    if n == 0:
        raise ValueError("no voxel above the Otsu threshold")
    tissue = labels == (np.argmax(np.bincount(labels.ravel())[1:]) + 1)
    # Fills only holes enclosed on every side (as MATLAB imfill "holes"): a calyx opening
    # that reaches the outside stays outside the fruit.
    fruit = ndimage.binary_fill_holes(tissue)
    air = fruit & ~tissue

    # Local mean of the tissue only: sum of tissue intensities in the window divided by the
    # number of tissue voxels in it. Air in the window is not counted, so tissue under the
    # skin is compared with tissue, not pulled towards the air outside.
    # UNVERIFIED: the paper uses MATLAB's adaptive threshold and gives no sensitivity; "below
    # the plain local mean" is our reading of "based on the local mean intensity".
    tissue_float = tissue.astype(np.float32)
    summed = ndimage.uniform_filter(volume * tissue_float, size=window, mode="constant")
    counted = ndimage.uniform_filter(tissue_float, size=window, mode="constant")
    local_mean = np.divide(
        summed, counted, out=np.zeros_like(summed), where=counted > 0
    )
    low = tissue & (volume < local_mean)
    high = tissue & ~low
    return {
        "tissue": tissue,
        "fruit": fruit,
        "air": air,
        "low_density": low,
        "high_density": high,
        "otsu_threshold": threshold,
    }


def tissue_features(scaled: np.ndarray, window: int = WINDOW) -> dict[str, float]:
    """The 10 features of one fruit (names in `FEATURE_NAMES`), plus the Otsu threshold."""
    from scipy.stats import ks_2samp

    volume = np.asarray(scaled, dtype=np.float32)
    masks = tissue_masks(volume, window)
    out = {"air_fraction": float(masks["air"].sum() / masks["fruit"].sum())}
    for name, key in [
        ("fruit", "fruit"),
        ("tissue", "tissue"),
        ("low_density", "low_density"),
        ("high_density", "high_density"),
    ]:
        values = volume[masks[key]]
        out[f"{name}_mean"] = float(values.mean())
        out[f"{name}_sd"] = float(values.std())
    out["ks_low_high"] = float(
        ks_2samp(volume[masks["low_density"]], volume[masks["high_density"]]).statistic
    )
    out["otsu_threshold"] = masks["otsu_threshold"]
    return out


def main() -> None:
    """Measure every open fruit of `ENCODING_RUN`; write a parquet table with run.json."""
    import nibabel as nib
    import polars as pl

    from pearlatent.config import load_config
    from pearlatent.pooling import scale_minmax
    from pearlatent.runs import new_run_dir, write_run_record

    cfg = load_config()
    run = cfg.path("derived_dir") / ENCODING_RUN_DIR
    # Open fruit only: the sealed test fruit are not in this manifest and stay unopened.
    manifest = pl.read_parquet(run / "manifest.parquet")
    print(f"open fruit in {ENCODING_RUN}: {manifest.height}")

    rows = []
    for k, fruit in enumerate(manifest.iter_rows(named=True)):
        path = cfg.path(f"ct_dir_{fruit['season']}") / f"{fruit['file_stem']}.nii"
        scaled = scale_minmax(np.asarray(nib.load(path).dataobj))
        rows.append({"fruit_id": fruit["fruit_id"], **tissue_features(scaled)})
        if (k + 1) % 50 == 0:
            print(f"  {k + 1} of {manifest.height}", flush=True)

    table = pl.DataFrame(rows)
    out_dir = new_run_dir(
        cfg.path("derived_dir") / "tissue_features", "tissue_features"
    )
    table.write_parquet(out_dir / "tissue_features.parquet")
    write_run_record(
        out_dir,
        {
            "encoding_run": ENCODING_RUN,
            "fruit": table.height,
            "window": WINDOW,
            "source": "Van De Looverbosch et al. 2020, Food Control 113, 107170",
        },
    )
    print(f"wrote {table.height} fruit to {out_dir}")


if __name__ == "__main__":
    main()

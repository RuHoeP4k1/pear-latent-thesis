"""Med3D (MedicalNet) ResNet-18 features of the 128³ volumes (2 November 2026).

Source: Chen, S., Ma, K. & Zheng, Y. (2019). Med3D: Transfer learning for 3D medical image
analysis. arXiv 1904.00625. Code: github.com/Tencent/MedicalNet at commit 20f76aa (MIT
licence), a separate clone (`med3d_repo` in `config/local.toml`), imported, not copied. Weights:
`resnet_18_23dataset.pth` from Hugging Face `TencentMedicalNet/MedicalNet-Resnet18`, SHA-256
61224f9317fcce873366deb3703183e92cc47325b726b69691b33536244e10f4 (checked by `main`).

A network pretrained on 23 medical CT and MRI datasets, used frozen: a comparison row for
Hugo's encoder, which was trained on pears. Choices (Ruben, 2 November 2026):

1. Input normalisation as MedicalNet's training (`datasets/brains18.py`): minus the mean,
   divided by the standard deviation of the object voxels. Object = voxels above 0.1 of the
   min-max scaled volume (the body threshold of Step 0); MedicalNet uses "above 0", but our
   background after min-max scaling is not exactly 0. This removes each fruit's mean and SD,
   so these features carry the pattern inside the fruit, not its overall brightness.
   Background (outside the fruit) set to 0 (Ruben, 2 November 2026). MedicalNet fills it
   with random noise of mean 0 and SD 1; a first run without noise left it at about -6.5
   (minus the fruit mean over the fruit SD), far from anything the network saw in training
   (figure `derived/figures/12_med3d_input.png`). 0 is the level of MedicalNet's noise, the
   value of the zero padding in every MedicalNet convolution, and the rule of nnU-Net
   (Isensee et al. 2021, Nature Methods 18, 203-211; github.com/MIC-DKFZ/nnUNet, branch
   nnunetv1, commit db16c6c, `nnunet/preprocessing/preprocessing.py` lines 302-305: z-score
   inside the mask, 0 outside). No noise, so the features are the same on every run.
   The fruit is the object voxels, largest connected part, enclosed holes filled (as the
   `fruit` mask of `pearlatent.tissue_features`), so cavities and core air keep their own
   low value; a first version set every voxel at or below 0.1 to 0, which made a cavity look
   like average tissue. The mean and SD still come from the object voxels only, so the size
   of a cavity does not change the scaling of the rest of the fruit.
2. Features: the backbone up to `layer4` (512 channels × 16³ for a 128³ input; the segmentation
   head is not used), averaged over the 16³ grid: 512 numbers per fruit.
3. Open fruit only (the manifest of `ENCODING_RUN`); the sealed fruit wait for the test.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

from pearlatent.runs import ENCODING_RUN, ENCODING_RUN_DIR

WEIGHTS_SHA256 = "61224f9317fcce873366deb3703183e92cc47325b726b69691b33536244e10f4"
OBJECT_THRESHOLD = 0.1  # min-max scaled intensity; the body threshold of Step 0
N_FEATURES = 512


def fruit_mask(scaled: np.ndarray) -> np.ndarray:
    """Voxels above `OBJECT_THRESHOLD`, largest connected part, enclosed holes filled.

    Out: a boolean array, the shape of `scaled`. Cavities and core air enclosed by tissue are
    inside the fruit; an opening that reaches the outside (calyx) stays outside.
    """
    from scipy import ndimage

    labels, n = ndimage.label(np.asarray(scaled) > OBJECT_THRESHOLD)
    if n == 0:
        raise ValueError("no voxel above the object threshold")
    largest = labels == (np.argmax(np.bincount(labels.ravel())[1:]) + 1)
    return ndimage.binary_fill_holes(largest)


def normalise_as_medicalnet(scaled: np.ndarray) -> np.ndarray:
    """Z-score a min-max scaled volume on its object voxels; 0 outside the fruit.

    In: a volume in [0, 1]. Out: float32, same shape. Mean and SD come from the object voxels
    (> `OBJECT_THRESHOLD`) inside the fruit, which then have mean 0 and SD 1; enclosed air
    keeps its (low) z-score; everything outside `fruit_mask` is 0 (as nnU-Net; no noise).
    """
    volume = np.asarray(scaled, dtype=np.float32)
    fruit = fruit_mask(volume)
    object_voxels = volume[fruit & (volume > OBJECT_THRESHOLD)]
    out = (volume - object_voxels.mean()) / object_voxels.std()
    out[~fruit] = 0.0
    return out


def load_backbone(weights: Path, repo: Path):
    """MedicalNet ResNet-18 with the Med3D weights, up to `layer4`, in evaluation mode.

    BatchNorm in evaluation mode uses the running statistics stored in the weights, so (unlike
    Hugo's InstanceNorm) a cell does not depend on the statistics of the whole fruit.
    """
    import torch

    sys.path.insert(0, str(repo))
    from models.resnet import resnet18  # MedicalNet's own definition (pinned clone)

    # shortcut_type "A" and these sizes as in MedicalNet's README for resnet_18_23dataset.
    model = resnet18(
        sample_input_D=128,
        sample_input_H=128,
        sample_input_W=128,
        num_seg_classes=1,
        shortcut_type="A",
    )
    # weights_only=True: load tensors only, never code from the file.
    checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    # The weights were saved from nn.DataParallel, which prefixes every name with "module.".
    state = {k.removeprefix("module."): v for k, v in checkpoint["state_dict"].items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    # Only the segmentation head may be missing: it is not in the pretrained file.
    if unexpected or any(not k.startswith("conv_seg.") for k in missing):
        raise ValueError(
            f"weights do not fit: missing {missing}, unexpected {unexpected}"
        )
    backbone = torch.nn.Sequential(
        model.conv1,
        model.bn1,
        model.relu,
        model.maxpool,
        model.layer1,
        model.layer2,
        model.layer3,
        model.layer4,
    )
    return backbone.eval()


def pooled_features(backbone, normalised: np.ndarray) -> np.ndarray:
    """Mean over the grid of the `layer4` output: 512 float32 numbers for one volume."""
    import torch

    x = torch.from_numpy(np.ascontiguousarray(normalised, dtype=np.float32))[None, None]
    with torch.inference_mode():
        out = backbone(x)  # (1, 512, 16, 16, 16) for a 128³ volume
    return out.mean(dim=(2, 3, 4))[0].numpy().astype(np.float32)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    """Features of every open fruit of `ENCODING_RUN`; writes features.npy, manifest, run.json."""
    import nibabel as nib
    import polars as pl
    import torch

    from pearlatent.config import load_config
    from pearlatent.pooling import scale_minmax
    from pearlatent.runs import new_run_dir, write_run_record

    cfg = load_config()
    weights = cfg.path("med3d_weights")
    if _sha256(weights) != WEIGHTS_SHA256:
        raise ValueError(f"{weights} is not the published Med3D file")
    backbone = load_backbone(weights, cfg.path("med3d_repo"))
    # CPU, one fruit at a time (about 0.5 s per fruit on the Mac, 2 November 2026).
    torch.set_grad_enabled(False)

    run = cfg.path("derived_dir") / ENCODING_RUN_DIR
    # Open fruit only: the sealed test fruit are not in this manifest and stay unopened.
    manifest = pl.read_parquet(run / "manifest.parquet")
    print(f"open fruit in {ENCODING_RUN}: {manifest.height}")

    features = np.zeros((manifest.height, N_FEATURES), dtype=np.float32)
    for k, fruit in enumerate(manifest.iter_rows(named=True)):
        path = cfg.path(f"ct_dir_{fruit['season']}") / f"{fruit['file_stem']}.nii"
        scaled = scale_minmax(np.asarray(nib.load(path).dataobj))
        features[k] = pooled_features(backbone, normalise_as_medicalnet(scaled))
        if (k + 1) % 50 == 0:
            print(f"  {k + 1} of {manifest.height}", flush=True)

    out_dir = new_run_dir(cfg.path("derived_dir") / "med3d", "med3d_resnet18")
    np.save(out_dir / "features.npy", features)
    manifest.write_parquet(out_dir / "manifest.parquet")
    write_run_record(
        out_dir,
        {
            "encoding_run": ENCODING_RUN,
            "fruit": manifest.height,
            "weights": "resnet_18_23dataset.pth",
            "weights_sha256": WEIGHTS_SHA256,
            "medicalnet_commit": "20f76aa",
            "normalisation": f"z-score over voxels > {OBJECT_THRESHOLD}, no noise",
            "pooling": "mean of layer4 over the 16^3 grid",
        },
        root=cfg.root,
    )
    print(f"wrote {manifest.height} fruit to {out_dir}")


if __name__ == "__main__":
    main()

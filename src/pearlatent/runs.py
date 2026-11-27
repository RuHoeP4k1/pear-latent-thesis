"""Run records: every product in derived/ or results/ is traceable to code and parameters."""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

# The runs in use, as folders under `derived_dir` (28 October 2026). After a rerun (for example
# new latents after reprocessing without the stalk), change them here; every notebook and
# module reads them from here.
ENCODING_RUN = "20261013T171306Z_hugo_vae"
ENCODING_RUN_DIR = f"latents/hugo_vae/{ENCODING_RUN}"
RANDOM_INIT_RUN_DIR = "latents/random_init/20261020T120336Z_random_init"
RAW_CT_32_RUN_DIR = "raw_ct_32/20261005T142026Z_raw_ct_32"
DESCRIPTORS_RUN_DIR = "descriptors/20261023T155219Z_descriptors"
# Stretch factor and real size from Hugo's 2526 originals (27 November 2026, `pearlatent.real_size`).
REAL_SIZE_RUN_DIR = "real_size/20261127T125309Z_real_size"
# Which rows follow the stretch or real size; AUC within size thirds (27 November 2026,
# `pearlatent.stretch_check`, in results/; methods sections 21 and 22).
STRETCH_CHECK_RUN_DIR = "20261127T133436Z_stretch_check"
TISSUE_FEATURES_RUN_DIR = "tissue_features/20261102T151947Z_tissue_features"
MED3D_RUN_DIR = "med3d/20261102T185107Z_med3d_resnet18"
# logvar of the encoder for the open 2526 fruit and Hugo's 66 validation fruit (9 November
# 2026, `pearlatent.logvar_run`; code unchanged since 868af71, saved while notebook 15 was untracked).
LOGVAR_RUN_DIR = "logvar/20261109T171513Z_logvar"
# EPO models of notebook 22 (22 November 2026), in results/: cavity-free browning ≥ 2 at 4 LVs
# (the counterfactual direction `cavity_free_epo`) and binary_23 at 3 LVs (Ruben's choice).
EPO_RUN_DIRS = {
    "cavity_free": "20261122T152905Z_epo_cavity_free",
    "general": "20261122T152916Z_epo_general",
}

# PLS-DA on binary_23 with EPO on the measured stretch factor, latent rows only, LVs of the
# rows without EPO (27 November 2026, `pearlatent.epo_stretch_run`; methods section 23).
EPO_STRETCH_RUN_DIRS = {
    "hugo_vae": "20261127T134101Z_plsda_binary23_hugo_vae_epo_stretch",
    "hugo_vae_hr": "20261127T134116Z_plsda_binary23_hugo_vae_hr_epo_stretch",
    "hugo_vae_hr_sd": "20261127T134132Z_plsda_binary23_hugo_vae_hr_sd_epo_stretch",
}

# The PLS-DA runs on binary_23 in use (folders under `results_dir`, saved 3 November 2026 by
# notebook 08; `docs/methods_plsda.md`, section 7). Notebook 10 compares them.
PLSDA_BINARY23_RUN_DIRS = {
    "hugo_vae": "20261103T101426Z_plsda_binary23_hugo_vae",
    "ct_32": "20261103T101429Z_plsda_binary23_ct_32",
    "tissue_features": "20261103T100917Z_plsda_binary23_tissue_features",
    "med3d": "20261103T100914Z_plsda_binary23_med3d",
    # Height-radius rows, 3 LVs each (Ruben, 7 November 2026).
    "hugo_vae_hr": "20261107T165738Z_plsda_binary23_hugo_vae_hr",
    "ct_32_hr": "20261107T165755Z_plsda_binary23_ct_32_hr",
    # Height-radius bins with the SD of each bin, 3 LVs each (Ruben, 9 November 2026).
    "hugo_vae_hr_sd": "20261109T173414Z_plsda_binary23_hugo_vae_hr_sd",
    "ct_32_hr_sd": "20261109T173415Z_plsda_binary23_ct_32_hr_sd",
}


def git_state(root: Path | None = None) -> dict:
    """Commit hash and whether the working tree has uncommitted changes."""

    def _git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, check=False
        ).stdout.strip()

    return {
        "commit": _git("rev-parse", "HEAD") or None,
        "dirty": bool(_git("status", "--porcelain")),
    }


def new_run_dir(base: Path, name: str) -> Path:
    """Create `base/<UTC timestamp>_<name>/` and return it."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(base) / f"{stamp}_{name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_run_record(run_dir: Path, params: dict, root: Path | None = None) -> Path:
    """Write run.json with parameters, git state, host and library versions."""
    import numpy as np
    import polars as pl

    try:
        import torch

        torch_info = {"torch": torch.__version__, "cuda": torch.version.cuda}
    except ImportError:
        torch_info = {}

    # The model results depend on these too; read from the installed packages, not imported.
    model_libraries = {
        name: metadata.version(name)
        for name in ("scikit-learn", "scipy", "statsmodels")
    }
    record = {
        "created_utc": datetime.now(UTC).isoformat(),
        "host": platform.node(),
        "python": platform.python_version(),
        "git": git_state(root),
        "versions": {
            "numpy": np.__version__,
            "polars": pl.__version__,
            **model_libraries,
            **torch_info,
        },
        "params": params,
    }
    out = Path(run_dir) / "run.json"
    out.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    return out


def latest_run_dir(base: Path, name: str) -> Path | None:
    """The newest `base/<timestamp>_<name>/` that contains a run.json, or None."""
    candidates = sorted(
        p
        for p in Path(base).glob(f"*_{name}")
        if p.is_dir() and (p / "run.json").exists()
    )
    return candidates[-1] if candidates else None

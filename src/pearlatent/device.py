"""Device selection and reproducibility for PyTorch."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DeviceInfo:
    device: str
    name: str
    total_memory_gb: float | None
    torch_version: str
    cuda_version: str | None
    bf16_supported: bool


def get_device(preference: str = "auto") -> str:
    """Return "cuda", "mps" or "cpu". `preference` overrides "auto" when available."""
    import torch

    if preference != "auto":
        return preference
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def describe_device(preference: str = "auto") -> DeviceInfo:
    import torch

    device = get_device(preference)
    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        return DeviceInfo(
            device=device,
            name=props.name,
            total_memory_gb=round(props.total_memory / 1024**3, 1),
            torch_version=torch.__version__,
            cuda_version=torch.version.cuda,
            bf16_supported=torch.cuda.is_bf16_supported(),
        )
    return DeviceInfo(
        device=device,
        name=device.upper(),
        total_memory_gb=None,
        torch_version=torch.__version__,
        cuda_version=None,
        bf16_supported=False,
    )


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch. `deterministic=True` trades speed for bitwise repeatability."""
    import torch

    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - seeds libraries that use the global generator
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False

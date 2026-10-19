"""Seed, library versions and hardware recorded with every experiment (decision D11)."""

import platform
import random
from importlib.metadata import PackageNotFoundError, version

import numpy as np
import torch

RECORDED_PACKAGES: tuple[str, ...] = (
    "torch",
    "torchvision",
    "facenet-pytorch",
    "scikit-learn",
    "numpy",
)


def seed_everything(seed: int) -> None:
    """Seeds Python, NumPy and PyTorch and turns off cuDNN autotuning."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def library_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for package in RECORDED_PACKAGES:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def hardware_info(device: torch.device) -> dict[str, str | None]:
    """Describes where the experiment ran, without host or user names."""
    on_cuda = device.type == "cuda"
    return {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if on_cuda else None,
        "cuda": torch.version.cuda if on_cuda else None,
        "os": platform.platform(terse=True),
        "cpu": platform.processor() or platform.machine(),
    }

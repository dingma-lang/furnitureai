from __future__ import annotations

import importlib.metadata
import platform
import sys
from time import perf_counter


def get_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError('PyTorch is missing; install ".[models]"') from exc
    return torch


def resolve_device(requested: str):
    torch = get_torch()
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if requested == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS requested but unavailable")
    return torch.device(requested)


def synchronize(device):
    torch = get_torch()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


class ModelTimer:
    """Synchronized elapsed seconds; excludes model loading when entered after load()."""

    def __init__(self, device):
        self.device = device
        self.seconds = None

    def __enter__(self):
        synchronize(self.device)
        self.start = perf_counter()
        return self

    def __exit__(self, *_):
        synchronize(self.device)
        self.seconds = perf_counter() - self.start


def environment():
    versions = {}
    for name in ("torch", "transformers", "faiss-cpu", "numpy", "Pillow"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": sys.version.split()[0], "platform": platform.platform(), "packages": versions}

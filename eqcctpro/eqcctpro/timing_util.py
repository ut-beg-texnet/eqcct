"""
Shared high-resolution timing for EQCCTPro trials and benchmarks.

Use :func:`monotonic_s` for all duration measurements (wall elapsed time) so results
are not affected by system clock adjustments (NTP). This matches common practice for
benchmarking and aligns Ripper, Model-Actor, and driver-side serial baselines.

CUDA: PyTorch kernels are often asynchronous. :func:`cuda_synchronize_best_effort`
blocks until the default CUDA device finishes queued work so stopwatches reflect
completed GPU inference/transfer when timing GPU SeisBench paths.
"""
from __future__ import annotations

import time


def monotonic_s() -> float:
    """Monotonic, high-resolution seconds (``time.perf_counter``)."""
    return time.perf_counter()


def cuda_synchronize_best_effort() -> None:
    """``torch.cuda.synchronize()`` when PyTorch CUDA is available; no-op otherwise."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass

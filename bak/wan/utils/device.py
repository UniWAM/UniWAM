"""Device helpers so that inference runs on CUDA, Intel XPU or CPU."""
import torch


def get_default_device():
    """CUDA when available, then Intel XPU, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


def device_autocast(device_type, dtype):
    """torch.autocast for device_type that also accepts float32.

    CUDA autocast takes float32 as its target dtype. Autocast on XPU and CPU
    only takes half precision types, so there a float32 region turns autocast
    off: float32 inputs and float32 weights then stay in float32, which is what
    the CUDA float32 region computes.
    """
    if device_type == "cuda" or dtype != torch.float32:
        return torch.amp.autocast(device_type, dtype=dtype)
    return torch.amp.autocast(device_type, enabled=False)

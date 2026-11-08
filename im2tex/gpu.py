import torch


def cap_memory(margin_gb: float = 0.5) -> float:
    """Limit PyTorch to the GPU memory free right now, minus a margin, so running out raises an OOM error.

    On Windows/WSL the NVIDIA driver otherwise spills into shared system RAM: ~10x slower, no error, and it
    can evict the desktop's graphics memory. Call before allocating anything. Returns the cap in GB."""
    free, total = torch.cuda.mem_get_info()
    cap = max(free - margin_gb * 2**30, 0)
    torch.cuda.set_per_process_memory_fraction(cap / total)
    return cap / 2**30

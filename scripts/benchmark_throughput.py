"""Measure training throughput (samples/s) and peak GPU memory of Im2Tex on this machine.

    python scripts/benchmark_throughput.py [--compile]
"""
import argparse
import time

import torch
import torch.nn.functional as F

from im2tex.model import Im2Tex

VOCAB = 8000
# (height, width, tokens): roughly a small, median, large and very large formula
SHAPES = [(64, 320, 48), (96, 480, 96), (128, 640, 160), (192, 896, 256)]
BATCH_SIZES = [128, 64, 32, 16, 8]


def step(model, opt, images, tokens):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(images, tokens[:, :-1])
    loss = F.cross_entropy(logits.flatten(0, 1).float(), tokens[:, 1:].flatten(), label_smoothing=0.1)
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)


def bench(model, opt, bsz, h, w, t, steps=20, warmup=5):
    images = torch.randn(bsz, 1, h, w, device="cuda").contiguous(memory_format=torch.channels_last)
    tokens = torch.randint(0, VOCAB, (bsz, t + 1), device="cuda")
    torch.cuda.reset_peak_memory_stats()
    for _ in range(warmup):
        step(model, opt, images, tokens)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        step(model, opt, images, tokens)
    torch.cuda.synchronize()
    return bsz * steps / (time.perf_counter() - t0), torch.cuda.max_memory_allocated() / 2**30


ap = argparse.ArgumentParser()
ap.add_argument("--compile", action="store_true")
args = ap.parse_args()

# On Windows/WSL the NVIDIA driver spills into shared system RAM instead of raising OOM, which makes
# oversized batches ~10x slower without any error. Capping the allocator turns that into a real OOM.
torch.cuda.set_per_process_memory_fraction(0.9)
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
model = Im2Tex(VOCAB, pretrained=False).cuda().to(memory_format=torch.channels_last)
print(f"{torch.cuda.get_device_name(0)} | params {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M"
      f" | compile={args.compile}")
opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
train_model = torch.compile(model) if args.compile else model

print(f"{'image':>10} {'tokens':>6} {'batch':>5} {'samples/s':>10} {'peak GB':>8}")
for h, w, t in SHAPES:
    for bsz in BATCH_SIZES:
        try:
            sps, mem = bench(train_model, opt, bsz, h, w, t)
        except torch.OutOfMemoryError:
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            continue
        print(f"{h:>4}x{w:<5} {t:>6} {bsz:>5} {sps:>10.0f} {mem:>8.2f}", flush=True)
        break

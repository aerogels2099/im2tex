"""Train the Im2Tex model on the shards (run scripts/index_shards.py first).

Resumable: re-running with the same --run continues from <run>/last.pt. Logs go to <run>/log.jsonl.
    python scripts/train.py --run base --steps 100000 [--subset 200000] [--compile]
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from torch.utils.data import DataLoader

from im2tex.data import PAD_ID, FormulaDataset, SizeBatchSampler, collate
from im2tex.gpu import cap_memory
from im2tex.model import Im2Tex, load_checkpoint, save_checkpoint
from im2tex.tokenizer import BOS, EOS, brace_delta, load

DATA = Path(os.environ.get("IM2TEX_DATA", Path.home() / "im2tex-data"))


def planned_sizes(index, max_h, max_w):
    """(N, 3): height and width after fitting, rounded up to multiples of 16, and token count."""
    w, h = index[:, 2] - index[:, 0], index[:, 3] - index[:, 1]
    s = np.minimum(np.minimum(max_h / h, max_w / w), 2.0)
    ceil16 = lambda x: (np.ceil(np.maximum(np.round(x), 1) / 16) * 16).astype(np.int64)  # noqa: E731
    return np.column_stack([ceil16(h * s), ceil16(w * s), index[:, 4]])


def lr_at(step, args):
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    progress = min(1.0, (step - args.warmup) / max(1, args.steps - args.warmup))
    return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))  # cosine down to 10%


@torch.no_grad()
def validate(model, batches, dataset, tok, delta):
    """Teacher-forced loss and greedy exact match on a fixed validation subset."""
    model.eval()
    loss_sum = tokens_sum = exact = total = 0
    for batch in batches:
        images, tokens = collate([dataset[i] for i in batch])
        images, tokens = images.cuda(), tokens.cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(images, tokens[:, :-1])
            out = model.generate(images, tok.token_to_id(BOS), tok.token_to_id(EOS),
                                 max_len=tokens.shape[1] + 32, brace_delta=delta)
        n = (tokens[:, 1:] != PAD_ID).sum().item()
        loss_sum += F.cross_entropy(logits.float().transpose(1, 2), tokens[:, 1:], ignore_index=PAD_ID,
                                    reduction="sum").item()
        tokens_sum += n
        preds, refs = tok.decode_batch(out.tolist()), tok.decode_batch(tokens.tolist())
        exact += sum(p == r for p, r in zip(preds, refs))
        total += len(batch)
    model.train()
    return {"val_loss": loss_sum / tokens_sum, "val_exact": exact / total}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run name; checkpoints and logs go to <data>/runs/<run>")
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--subset", type=int, help="train on a fixed random subset of this many rows")
    ap.add_argument("--max-h", type=int, default=256)
    ap.add_argument("--max-w", type=int, default=1024)
    ap.add_argument("--max-tokens", type=int, default=512, help="longest label (decoder positions)")
    ap.add_argument("--max-pixels", type=int, default=2_500_000, help="per batch, after padding")
    ap.add_argument("--max-batch-tokens", type=int, default=8192)
    ap.add_argument("--max-batch", type=int, default=256)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--encoder", default="convnext_tiny")
    ap.add_argument("--val-size", type=int, default=1000)
    ap.add_argument("--val-every", type=int, default=2000)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run = DATA / "runs" / args.run
    run.mkdir(parents=True, exist_ok=True)
    print(f"memory cap {cap_memory():.1f} GB")
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    # cuDNN builds kernels for every new input shape (~4 s per step here), and size-bucketed batches bring a new
    # shape almost every step; PyTorch's own kernels are ~30% slower on a repeated shape but cost nothing per shape.
    torch.backends.cudnn.enabled = False
    tok = load(DATA / "tokenizer.json")
    delta = torch.tensor(brace_delta(tok), device="cuda")

    shards = DATA / "shards"
    train_rows, val_rows = load_from_disk(str(shards / "train")), load_from_disk(str(shards / "val"))
    train_index, val_index = np.load(shards / "train" / "index.npy"), np.load(shards / "val" / "index.npy")
    train_sizes = planned_sizes(train_index, args.max_h, args.max_w)
    keep = np.flatnonzero(train_sizes[:, 2] <= args.max_tokens)
    if args.subset:
        keep = np.sort(np.random.default_rng(0).permutation(keep)[: args.subset])
    train_set = FormulaDataset(train_rows, train_index, tok, args.max_h, args.max_w, train=True)
    val_set = FormulaDataset(val_rows, val_index, tok, args.max_h, args.max_w, train=False)
    val_sizes = planned_sizes(val_index, args.max_h, args.max_w)
    val_keep = np.flatnonzero(val_sizes[:, 2] <= args.max_tokens)
    val_pick = np.random.default_rng(0).choice(val_keep, min(args.val_size, len(val_keep)), replace=False)
    budget = dict(max_pixels=args.max_pixels, max_tokens=args.max_batch_tokens, max_batch=args.max_batch)
    val_batches = list(SizeBatchSampler(val_sizes, val_pick, **budget).batches(np.sort(val_pick)))
    print(f"train rows {len(keep)} (of {len(train_rows)}), val subset {len(val_pick)}")

    if (run / "last.pt").exists():
        model, ckpt = load_checkpoint(run / "last.pt")
        model.train()
        step = ckpt["step"]
        print(f"resuming {run.name} at step {step}")
    else:
        model = Im2Tex(tok.get_vocab_size(), d_model=args.d_model, n_layers=args.layers,
                       encoder=args.encoder, max_tokens=args.max_tokens).cuda()
        ckpt, step = None, 0
    model = model.to(memory_format=torch.channels_last)
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=args.lr, fused=True)
    if ckpt:
        opt.load_state_dict(ckpt["opt"])
    best = ckpt.get("best", 0.0) if ckpt else 0.0
    train_model = torch.compile(model, dynamic=True) if args.compile else model
    (run / "args.json").write_text(json.dumps(vars(args), indent=2))

    sampler = SizeBatchSampler(train_sizes, keep, seed=args.seed + step, **budget)
    loader = DataLoader(train_set, batch_sampler=sampler, collate_fn=collate, num_workers=args.workers,
                        pin_memory=True, persistent_workers=True, prefetch_factor=4)
    log = (run / "log.jsonl").open("a")
    t0, images_seen, tokens_seen = time.time(), 0, 0
    for images, tokens in loader:
        if step >= args.steps:
            break
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args)
        images = images.cuda(non_blocking=True).to(memory_format=torch.channels_last)
        tokens = tokens.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = train_model(images, tokens[:, :-1])
        loss = F.cross_entropy(logits.float().transpose(1, 2), tokens[:, 1:], ignore_index=PAD_ID,
                               label_smoothing=0.1)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        images_seen += len(images)
        tokens_seen += (tokens[:, 1:] != PAD_ID).sum().item()

        entry = {}
        if step % args.log_every == 0:
            dt = time.time() - t0
            entry |= {"loss": round(loss.item(), 4), "grad_norm": round(grad_norm.item(), 3),
                      "lr": lr_at(step, args), "img_per_s": round(images_seen / dt, 1),
                      "tok_per_s": round(tokens_seen / dt), "batch": len(images),
                      "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
        if step % args.val_every == 0 or step == args.steps:
            entry |= validate(model, val_batches, val_set, tok, delta)
            if entry["val_exact"] > best or not (run / "best.pt").exists():
                best = entry["val_exact"]
                save_checkpoint(model, run / "best.pt", step=step, val_exact=best)
        if step % args.ckpt_every == 0 or step == args.steps:
            save_checkpoint(model, run / "last.pt", step=step, opt=opt.state_dict(), best=best)
        if entry:
            entry = {"step": step, **entry}
            print(json.dumps(entry), flush=True)
            log.write(json.dumps(entry) + "\n")
            log.flush()
            t0, images_seen, tokens_seen = time.time(), 0, 0  # throughput is per log window


if __name__ == "__main__":
    main()

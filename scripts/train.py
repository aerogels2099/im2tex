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
from rapidfuzz.distance import Levenshtein
from torch.utils.data import DataLoader

from im2tex.augment import draw_scales
from im2tex.data import LINE_SOURCES, PAD_ID, FormulaDataset, SizeBatchSampler, collate, planned_sizes
from im2tex.gpu import cap_memory
from im2tex.model import Im2Tex, load_checkpoint, save_checkpoint
from im2tex.tokenizer import BOS, EOS, LINE, brace_delta, load

DATA = Path(os.environ.get("IM2TEX_DATA", Path.home() / "im2tex-data"))


def lr_at(step, args):
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    progress = min(1.0, (step - args.warmup) / max(1, args.steps - args.warmup))
    return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))  # cosine down to 10%


@torch.no_grad()
def validate(model, batches, dataset, tok, delta, start):
    """-> teacher-forced loss per token, greedy readings from `start` (<bos> or <line>), and their references."""
    model.eval()
    loss_sum = tokens_sum = 0
    preds, refs = [], []
    for batch in batches:
        images, tokens = collate([dataset[i] for i in batch])
        images, tokens = images.cuda(), tokens.cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(images, tokens[:, :-1])
            out = model.generate(images, start, tok.token_to_id(EOS), max_len=tokens.shape[1] + 32, brace_delta=delta)
        loss_sum += F.cross_entropy(logits.float().transpose(1, 2), tokens[:, 1:], ignore_index=PAD_ID,
                                    reduction="sum").item()
        tokens_sum += (tokens[:, 1:] != PAD_ID).sum().item()
        preds += tok.decode_batch(out.tolist())
        refs += tok.decode_batch(tokens.tolist())
    model.train()
    return loss_sum / tokens_sum, preds, refs


def cer(preds, refs):
    """Character error rate: edits over reference characters, over the whole set."""
    return sum(Levenshtein.distance(p, r) for p, r in zip(preds, refs)) / max(sum(len(r) for r in refs), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run name; checkpoints and logs go to <data>/runs/<run>")
    ap.add_argument("--init", help="start from this checkpoint's weights (its architecture; new optimizer and schedule)")
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--subset", type=int, help="train on a fixed random subset of this many rows")
    ap.add_argument("--repeat", nargs="*", default=[], metavar="SOURCE=K",
                    help="show each row of SOURCE K times per pass over the data, e.g. iam=30")
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
    ap.add_argument("--val-lines", type=int, default=1000, help="validation rows from the line sources")
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
    # One size per row for the whole run (same seed on resume), so the sampler can plan batches with it.
    train_sources = np.asarray(train_rows["source"])
    scales = draw_scales(train_sources, np.random.default_rng(args.seed))
    train_sizes = planned_sizes(train_index, args.max_h, args.max_w, scales)
    keep = np.flatnonzero(train_sizes[:, 2] <= args.max_tokens)
    if args.subset:
        keep = np.sort(np.random.default_rng(0).permutation(keep)[: args.subset])
    for source, k in (r.split("=") for r in args.repeat):
        keep = np.concatenate([keep, *[keep[train_sources[keep] == source]] * (int(k) - 1)])
    train_set = FormulaDataset(train_rows, train_index, tok, args.max_h, args.max_w, train=True, scales=scales)
    val_set = FormulaDataset(val_rows, val_index, tok, args.max_h, args.max_w, train=False)
    val_sizes = planned_sizes(val_index, args.max_h, args.max_w)
    # Formulas and lines are validated apart, so val_exact stays comparable with runs that had no lines.
    val_sources = np.asarray(val_rows["source"])
    is_line, fits = np.isin(val_sources, list(LINE_SOURCES)), val_sizes[:, 2] <= args.max_tokens

    def pick(rows, n):
        return np.random.default_rng(0).choice(rows, min(n, len(rows)), replace=False)

    val_pick = pick(np.flatnonzero(fits & ~is_line), args.val_size)
    line_pick = pick(np.flatnonzero(fits & is_line), args.val_lines)
    budget = dict(max_pixels=args.max_pixels, max_tokens=args.max_batch_tokens, max_batch=args.max_batch)
    val_batches = list(SizeBatchSampler(val_sizes, val_pick, **budget).batches(np.sort(val_pick)))
    line_batches = list(SizeBatchSampler(val_sizes, line_pick, **budget).batches(np.sort(line_pick)))
    print(f"train rows {len(keep)} (of {len(train_rows)}), val {len(val_pick)} formulas + {len(line_pick)} lines")

    if (run / "last.pt").exists():
        model, ckpt = load_checkpoint(run / "last.pt")
        model.train()
        step = ckpt["step"]
        print(f"resuming {run.name} at step {step}")
    elif args.init:
        model, _ = load_checkpoint(args.init)
        if model.config["vocab_size"] < tok.get_vocab_size():  # tokens added since, e.g. <line>
            print(f"vocabulary {model.config['vocab_size']} -> {tok.get_vocab_size()}, new tokens start as <bos>")
            model.grow_vocab(tok.get_vocab_size(), tok.token_to_id(BOS))
        model.train()
        ckpt, step = None, 0
        print(f"starting from the weights in {args.init}")
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
    t0, images_seen, tokens_seen = time.monotonic(), 0, 0
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
            dt = time.monotonic() - t0
            entry |= {"loss": round(loss.item(), 4), "grad_norm": round(grad_norm.item(), 3),
                      "lr": lr_at(step, args), "img_per_s": round(images_seen / dt, 1),
                      "tok_per_s": round(tokens_seen / dt), "batch": len(images),
                      "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
        if step % args.val_every == 0 or step == args.steps:
            loss, preds, refs = validate(model, val_batches, val_set, tok, delta, tok.token_to_id(BOS))
            entry |= {"val_loss": loss, "val_exact": float(np.mean([p == r for p, r in zip(preds, refs)]))}
            score = entry["val_exact"]
            if line_batches:
                loss, preds, refs = validate(model, line_batches, val_set, tok, delta, tok.token_to_id(LINE))
                sources = val_sources[np.concatenate(line_batches)]
                entry |= {"val_line_loss": loss, "val_cer": cer(preds, refs)}
                for s in np.unique(sources):
                    entry[f"val_cer_{s}"] = cer(*zip(*[(p, r) for p, r, x in zip(preds, refs, sources) if x == s]))
                score -= entry["val_cer"]  # formulas right and lines read well
            if score > best or not (run / "best.pt").exists():
                best = score
                save_checkpoint(model, run / "best.pt", step=step, val_exact=entry["val_exact"], score=score)
        if step % args.ckpt_every == 0 or step == args.steps:
            save_checkpoint(model, run / "last.pt", step=step, opt=opt.state_dict(), best=best)
        if entry:
            entry = {"step": step, **entry}
            print(json.dumps(entry), flush=True)
            log.write(json.dumps(entry) + "\n")
            log.flush()
            t0, images_seen, tokens_seen = time.monotonic(), 0, 0  # throughput is per log window


if __name__ == "__main__":
    main()

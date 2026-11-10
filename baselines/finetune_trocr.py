"""Fine-tune TrOCR-small on the training shards: the "general OCR model adapted to LaTeX" baseline.

TrOCR squashes every image to 384x384 and keeps its own text tokenizer; that is the point of the comparison.
Resumable: re-running continues from the last checkpoint in --out.
    python baselines/finetune_trocr.py --steps 60000
"""
import argparse
import io
import math
import os
import random
import time
from pathlib import Path

import torch
from datasets import load_from_disk
from PIL import Image
from torch.utils.data import DataLoader, Dataset, RandomSampler
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from im2tex.gpu import cap_memory

DATA = Path(os.environ.get("IM2TEX_DATA", Path.home() / "im2tex-data"))


class Rows(Dataset):
    def __init__(self, ds, processor, max_len):
        self.ds, self.processor, self.max_len = ds, processor, max_len

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        r = self.ds[i]
        image = Image.open(io.BytesIO(r["image"])).convert("RGB")
        pixels = self.processor(images=image, return_tensors="pt").pixel_values[0]
        ids = self.processor.tokenizer(r["label"], truncation=True, max_length=self.max_len).input_ids
        return pixels, torch.tensor(ids)


def collate(batch):
    pixels = torch.stack([b[0] for b in batch])
    labels = torch.nn.utils.rnn.pad_sequence([b[1] for b in batch], batch_first=True, padding_value=-100)
    return pixels, labels


@torch.no_grad()
def val_loss(model, loader, batches=50):
    model.eval()
    losses = []
    for (pixels, labels), _ in zip(loader, range(batches)):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            losses.append(model(pixel_values=pixels.cuda(), labels=labels.cuda()).loss.item())
    model.train()
    return sum(losses) / len(losses)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="microsoft/trocr-small-printed")
    ap.add_argument("--out", type=Path, default=DATA / "checkpoints" / "trocr-small-ft")
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--micro", type=int, default=6, help="samples per forward pass: a batch of 12 with a "
                    "512-token label runs out of memory on 8 GB (the loss holds fp32 logits over a 64k vocab)")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max-len", type=int, default=512, help="label tokens (TrOCR's own tokenizer)")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--every", type=int, default=2000, help="validate and checkpoint every N steps")
    args = ap.parse_args()

    print(f"memory cap {cap_memory():.1f} GB")
    torch.manual_seed(0)
    resume = (args.out / "state.pt").exists()
    source = str(args.out) if resume else args.base
    processor = TrOCRProcessor.from_pretrained(source)
    model = VisionEncoderDecoderModel.from_pretrained(source).cuda().train()
    # The loss shifts labels using config.decoder_start_token_id, which newer transformers only keeps in the
    # generation config; copy it over so training starts decoding from the same token as generate().
    model.config.decoder_start_token_id = model.generation_config.decoder_start_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id

    train = Rows(load_from_disk(str(DATA / "shards" / "train")), processor, args.max_len)
    val_ds = load_from_disk(str(DATA / "shards" / "val"))
    val = Rows(val_ds.select(random.Random(0).sample(range(len(val_ds)), 2000)), processor, args.max_len)
    loader = DataLoader(train, batch_size=args.batch, sampler=RandomSampler(train, replacement=True,
                        num_samples=args.steps * args.batch), collate_fn=collate, num_workers=args.workers,
                        drop_last=True, persistent_workers=True)
    val_loader = DataLoader(val, batch_size=args.batch, collate_fn=collate, num_workers=4)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    warmup = 1000
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warmup) * 0.5 *
                                              (1 + math.cos(math.pi * min(s, args.steps) / args.steps)))
    step = 0
    if resume:
        state = torch.load(args.out / "state.pt", map_location="cpu")
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        step = state["step"]
        print(f"resumed at step {step}")

    t0, seen = time.monotonic(), 0
    for pixels, labels in loader:
        if step >= args.steps:
            break
        tokens, batch_loss = (labels != -100).sum(), 0.0
        for px, lb in zip(pixels.split(args.micro), labels.split(args.micro)):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(pixel_values=px.cuda(non_blocking=True), labels=lb.cuda(non_blocking=True)).loss
            loss = loss * (lb != -100).sum() / tokens  # weighted by tokens: sums to the whole batch's mean loss
            loss.backward()
            batch_loss += loss.detach()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        seen += len(pixels)
        if step % 100 == 0:
            print(f"step {step} loss {batch_loss.item():.3f} lr {sched.get_last_lr()[0]:.2e} "
                  f"{seen / (time.monotonic() - t0):.0f} img/s", flush=True)
        if step % args.every == 0 or step == args.steps:
            print(f"step {step} val_loss {val_loss(model, val_loader):.3f}", flush=True)
            model.save_pretrained(args.out)
            processor.save_pretrained(args.out)
            torch.save({"opt": opt.state_dict(), "sched": sched.state_dict(), "step": step}, args.out / "state.pt")

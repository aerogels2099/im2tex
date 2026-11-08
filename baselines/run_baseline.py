"""Run a pretrained model over a prepared split and write its refs.jsonl rows back with "pred" and "seconds".

Each baseline may live in its own venv (pinned dependencies conflict); run from the repo root with
PYTHONPATH=. so im2tex is importable.
    ~/venvs/pix2tex/bin/python baselines/run_baseline.py pix2tex ~/im2tex-data/crohme/2019 runs/pix2tex/crohme2019.jsonl --limit 500
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from PIL import Image

from im2tex.gpu import cap_memory
from im2tex.normalize import strip_math_delimiters


def load_pix2tex():
    from munch import Munch
    from pix2tex.cli import LatexOCR

    # LatexOCR() with no arguments silently runs on CPU (no_cuda=True); these are its defaults otherwise,
    # except temperature: its default 0.2 samples, so repeated runs disagree. 1e-8 is effectively greedy.
    model = LatexOCR(Munch(config="settings/config.yaml", checkpoint="checkpoints/weights.pth",
                           no_cuda=False, no_resize=False, temperature=1e-8))
    return lambda images: [model(im) for im in images]


def load_got_ocr():
    from transformers import AutoModelForImageTextToText, AutoProcessor

    name = "stepfun-ai/GOT-OCR-2.0-hf"
    processor = AutoProcessor.from_pretrained(name)
    model = AutoModelForImageTextToText.from_pretrained(name, dtype=torch.bfloat16).cuda().eval()

    def predict(images):
        # format=True is GOT's formatted-output mode (LaTeX for math); plain mode returns unusable text
        inputs = processor(images, return_tensors="pt", format=True).to("cuda", dtype=torch.bfloat16)
        out = model.generate(**inputs, do_sample=False, tokenizer=processor.tokenizer,
                             stop_strings="<|im_end|>", max_new_tokens=1024)
        return processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    return predict


def load_texify():
    from texify.inference import batch_inference
    from texify.model.model import load_model
    from texify.model.processor import load_processor

    model, processor = load_model(), load_processor()
    return lambda images: batch_inference(images, model, processor)


MODELS = {"pix2tex": load_pix2tex, "got_ocr": load_got_ocr, "texify": load_texify}


def predict_all(predict, images):
    """Predict a batch; on GPU out-of-memory, split it in half and retry."""
    try:
        return predict(images)
    except torch.OutOfMemoryError:
        if len(images) == 1:
            raise
    # Retry outside the except block: until it exits, the traceback keeps the failed attempt's tensors alive.
    torch.cuda.empty_cache()
    mid = len(images) // 2
    return predict_all(predict, images[:mid]) + predict_all(predict, images[mid:])


def subset(rows, limit):
    """The `limit` rows with the smallest id hash: a fixed sample that every model gets."""
    return sorted(rows, key=lambda r: hashlib.md5(r["id"].encode()).hexdigest())[:limit] if limit else rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=MODELS)
    ap.add_argument("split_dir", type=Path, help="directory with refs.jsonl")
    ap.add_argument("out", type=Path)
    ap.add_argument("--limit", type=int, help="fixed subset of this many rows (same for every model)")
    ap.add_argument("--batch", type=int, default=1, help="GOT-OCR needs 3.4 GB at 1, 5.7 GB at 2")
    args = ap.parse_args()

    print(f"memory cap {cap_memory():.1f} GB")
    predict = MODELS[args.model]()
    rows = subset([json.loads(l) for l in (args.split_dir / "refs.jsonl").open(encoding="utf-8")], args.limit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as out:
        for i in range(0, len(rows), args.batch):
            batch = rows[i:i + args.batch]
            images = [Image.open(args.split_dir / r["image"]).convert("RGB") for r in batch]
            start = time.perf_counter()
            preds = predict_all(predict, images)
            seconds = (time.perf_counter() - start) / len(batch)
            for r, p in zip(batch, preds):
                pred = strip_math_delimiters(p.strip()).strip()
                out.write(json.dumps({**r, "pred": pred, "seconds": seconds}) + "\n")
    peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_initialized() else 0
    print(f"{len(rows)} predictions -> {args.out} (peak GPU memory {peak:.2f} GB)")

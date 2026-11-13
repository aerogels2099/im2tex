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


def load_unimernet():
    import os

    from omegaconf import OmegaConf

    import unimernet  # noqa: F401  (registers its models and processors)
    from unimernet.common.registry import registry
    from unimernet.processors import load_processor

    weights = os.path.join(os.environ.get("IM2TEX_DATA", os.path.expanduser("~/im2tex-data")),
                           "models", "unimernet_base")
    cfg = OmegaConf.create({  # the settings of the official demo's cfg_base.yaml
        "arch": "unimernet", "model_name": "unimernet", "tokenizer_name": "nougat",
        "model_config": {"model_name": weights, "max_seq_len": 1536},
        "tokenizer_config": {"path": weights}, "load_pretrained": False,
    })
    model = registry.get_model_class("unimernet").from_config(cfg)
    # Load strictly: the package's own loader uses strict=False and would keep random weights on a mismatch.
    model.load_state_dict(torch.load(os.path.join(weights, "pytorch_model.pth"), map_location="cpu")["model"])
    model = model.cuda().eval()
    processor = load_processor("formula_image_eval", OmegaConf.create({"image_size": [192, 672]}))

    def predict(images):
        batch = torch.stack([processor(im) for im in images]).cuda()
        return model.generate({"image": batch})["pred_str"]  # greedy, like the demo

    return predict


def load_trocr(checkpoint="microsoft/trocr-small-printed"):
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    processor = TrOCRProcessor.from_pretrained(checkpoint)
    model = VisionEncoderDecoderModel.from_pretrained(checkpoint).cuda().eval()

    def predict(images):
        pixels = processor(images=images, return_tensors="pt").pixel_values.cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.generate(pixels, max_new_tokens=512, num_beams=1)
        return processor.batch_decode(out, skip_special_tokens=True)

    return predict


def load_trocr_ft():
    import os
    return load_trocr(os.path.join(os.environ.get("IM2TEX_DATA", os.path.expanduser("~/im2tex-data")),
                                   "checkpoints", "trocr-small-ft"))


def load_im2tex(run, beams=1, both_sizes=True, lines=False):
    """Our model: the best checkpoint of training run <data>/runs/<run>."""
    from im2tex.inference import load_predictor

    return load_predictor(run, beams=beams, both_sizes=both_sizes, lines=lines)


MODELS = {"pix2tex": load_pix2tex, "got_ocr": load_got_ocr, "texify": load_texify, "unimernet": load_unimernet,
          "trocr": load_trocr, "trocr_ft": load_trocr_ft}


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
    ap.add_argument("model", help=f"one of {', '.join(MODELS)}, or im2tex-<run> for our own model")
    ap.add_argument("split_dir", type=Path, help="directory with refs.jsonl")
    ap.add_argument("out", type=Path)
    ap.add_argument("--limit", type=int, help="fixed subset of this many rows (same for every model)")
    ap.add_argument("--batch", type=int, default=1, help="GOT-OCR needs 3.4 GB at 1, 5.7 GB at 2")
    ap.add_argument("--beams", type=int, default=1, help="beam search width (our model only)")
    ap.add_argument("--one-size", action="store_true", help="read small images only enlarged (our model only)")
    ap.add_argument("--line", action="store_true",
                    help="images are lines of words and math, read as text with $...$ (our model, run4 on); "
                         "score with scripts/evaluate_pages.py")
    args = ap.parse_args()

    print(f"memory cap {cap_memory():.1f} GB")
    if args.model.startswith("im2tex-"):
        predict = load_im2tex(args.model.removeprefix("im2tex-"), args.beams, both_sizes=not args.one_size,
                              lines=args.line)
    elif args.beams > 1 or args.one_size or args.line:
        ap.error("--beams, --one-size and --line are only implemented for our model (im2tex-<run>)")
    elif args.model in MODELS:
        predict = MODELS[args.model]()
    else:
        ap.error(f"unknown model {args.model}")
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
                pred = p.strip() if args.line else strip_math_delimiters(p.strip()).strip()
                out.write(json.dumps({**r, "pred": pred, "seconds": seconds}) + "\n")
    peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_initialized() else 0
    print(f"{len(rows)} predictions -> {args.out} (peak GPU memory {peak:.2f} GB)")

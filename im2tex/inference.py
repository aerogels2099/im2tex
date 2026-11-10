"""Turn equation images into LaTeX with a trained run's best checkpoint."""
import json
import os
from pathlib import Path

import torch
from PIL import Image, ImageOps

from im2tex.data import collate, preprocess
from im2tex.model import load_checkpoint
from im2tex.tokenizer import BOS, EOS, brace_delta, load

DATA = Path(os.environ.get("IM2TEX_DATA", Path.home() / "im2tex-data"))


def load_image(path) -> Image.Image:
    """RGB image the way it looks in a viewer: phone photos turned upright, transparency on white."""
    im = ImageOps.exif_transpose(Image.open(path))
    if "A" in im.getbands():  # transparent pixels would otherwise turn black
        im = Image.alpha_composite(Image.new("RGBA", im.size, "white"), im.convert("RGBA"))
    return im.convert("RGB")


def load_predictor(run: str = "base", device: str = "cuda", beams: int = 1, both_sizes: bool = True):
    """-> predict(list of PIL images) -> list of LaTeX strings, using <data>/runs/<run>/best.pt.

    Images are enlarged up to 2x to fit, which suits renders like the training data but hurts real scans and
    documents. With both_sizes, an image that gets enlarged is also read at its own size, and where the two
    readings differ the one with the higher mean token log-probability is kept."""
    if device == "cuda":
        torch.backends.cudnn.enabled = False  # every image is a new shape; see scripts/train.py
    run_dir = DATA / "runs" / run
    args = json.loads((run_dir / "args.json").read_text())
    model, _ = load_checkpoint(run_dir / "best.pt", device)
    tok = load(DATA / "tokenizer.json")
    delta = torch.tensor(brace_delta(tok), device=device)
    autocast = dict(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda")

    def batch(xs):
        return collate([(x, torch.zeros(1)) for x in xs])[0].to(device)

    def read(xs):
        with torch.autocast(**autocast):
            out = model.generate(batch(xs), tok.token_to_id(BOS), tok.token_to_id(EOS), brace_delta=delta,
                                 beams=beams)
        return tok.decode_batch(out.tolist())

    @torch.no_grad()
    def mean_logp(x, latex):
        ids = torch.tensor([tok.encode(latex).ids[: model.config["max_tokens"]]], device=device)  # <bos> ... <eos>
        with torch.autocast(**autocast):
            logits = model(batch([x]), ids[:, :-1])
        return logits.float().log_softmax(-1).gather(-1, ids[:, 1:, None]).mean().item()

    def predict(images):
        big = [preprocess(im, args["max_h"], args["max_w"]) for im in images]
        out = read(big)
        if not both_sizes:
            return out
        small = [preprocess(im, args["max_h"], args["max_w"], max_up=1.0) for im in images]
        redo = [i for i, (a, b) in enumerate(zip(big, small)) if a.shape != b.shape]
        for i, alt in zip(redo, read([small[i] for i in redo]) if redo else []):
            if alt != out[i] and mean_logp(small[i], alt) > mean_logp(big[i], out[i]):
                out[i] = alt
        return out

    return predict

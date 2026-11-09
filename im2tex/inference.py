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


def load_predictor(run: str = "base", device: str = "cuda", beams: int = 1):
    """-> predict(list of PIL images) -> list of LaTeX strings, using <data>/runs/<run>/best.pt."""
    if device == "cuda":
        torch.backends.cudnn.enabled = False  # every image is a new shape; see scripts/train.py
    run_dir = DATA / "runs" / run
    args = json.loads((run_dir / "args.json").read_text())
    model, _ = load_checkpoint(run_dir / "best.pt", device)
    tok = load(DATA / "tokenizer.json")
    delta = torch.tensor(brace_delta(tok), device=device)

    def predict(images):
        x = collate([(preprocess(im, args["max_h"], args["max_w"]), torch.zeros(1)) for im in images])[0]
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            out = model.generate(x.to(device), tok.token_to_id(BOS), tok.token_to_id(EOS), brace_delta=delta,
                                 beams=beams)
        return tok.decode_batch(out.tolist())

    return predict

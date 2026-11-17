"""Turn equation images into LaTeX with a trained run's best checkpoint."""
import json
import os
from pathlib import Path

import torch
from PIL import Image, ImageOps

from im2tex.data import collate, preprocess
from im2tex.model import load_checkpoint
from im2tex.tokenizer import BOS, EOS, LINE, brace_delta, load

DATA = Path(os.environ.get("IM2TEX_DATA", Path.home() / "im2tex-data"))


def load_image(path) -> Image.Image:
    """RGB image the way it looks in a viewer: phone photos turned upright, transparency on white."""
    im = ImageOps.exif_transpose(Image.open(path))
    if "A" in im.getbands():  # transparent pixels would otherwise turn black
        im = Image.alpha_composite(Image.new("RGBA", im.size, "white"), im.convert("RGBA"))
    return im.convert("RGB")


def load_predictor(run: str = "run7", device: str = "cuda", beams: int = 1, both_sizes: bool = True,
                   lines: bool = False):
    """-> predict(list of PIL images) -> list of LaTeX strings, using <data>/runs/<run>/best.pt. With lines, each
    image is read as a line of words and math (im2tex/page.py format) instead of one formula.

    Images are enlarged up to 2x to fit, which suits renders like the training data but hurts real scans and
    documents. With both_sizes, an image that gets enlarged is also read at its own size, and where the two
    readings differ the one with the higher mean token log-probability is kept.

    A reading that starts repeating itself is ended after the first copy of the repeat (Im2Tex.generate's
    stop_loops), and where only one of the two readings did, the other is kept. A second reading more than
    2x + 20 tokens longer than a first that didn't repeat is looping in a way that check misses, so it is
    stopped there and dropped: the longest right one in the test sets was 2.5x but only 15 tokens longer."""
    if device == "cuda":
        torch.backends.cudnn.enabled = False  # every image is a new shape; see scripts/train.py
    run_dir = DATA / "runs" / run
    args = json.loads((run_dir / "args.json").read_text())
    model, _ = load_checkpoint(run_dir / "best.pt", device)
    tok = load(DATA / "tokenizer.json")
    vocab = model.config["vocab_size"]  # the tokenizer may have gained tokens since this run was trained
    delta = torch.tensor(brace_delta(tok)[:vocab], device=device)
    start, close = tok.token_to_id(LINE if lines else BOS), tok.token_to_id("}</w>")
    if start >= vocab:
        raise ValueError(f"{run} was trained before line reading existed")
    autocast = dict(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda")

    def batch(xs):
        return collate([(x, torch.zeros(1)) for x in xs])[0].to(device)

    def read(xs, max_len=None):
        """-> LaTeX strings, their token counts (without <bos> and <eos>) and whether each was stopped for
        repeating itself. A reading stopped early (repeating, or at max_len) gets its open braces closed."""
        eos = tok.token_to_id(EOS)
        with torch.autocast(**autocast):
            out, looped = model.generate(batch(xs), start, eos, max_len=max_len, brace_delta=delta, beams=beams,
                                         stop_loops=True)
        unclosed = delta[out].sum(1).clamp(min=0).tolist()
        ids = [row + [close] * d for row, d in zip(out.tolist(), unclosed)]
        return tok.decode_batch(ids), (out[:, 1:] != eos).sum(1).tolist(), looped.tolist()

    @torch.no_grad()
    def mean_logp(x, latex):
        ids = torch.tensor([tok.encode(latex).ids[: model.config["max_tokens"]]], device=device)  # <bos> ... <eos>
        ids[0, 0] = start
        with torch.autocast(**autocast):
            logits = model(batch([x]), ids[:, :-1])
        return logits.float().log_softmax(-1).gather(-1, ids[:, 1:, None]).mean().item()

    def predict(images):
        big = [preprocess(im, args["max_h"], args["max_w"]) for im in images]
        out, lengths, looped = read(big)
        if not both_sizes:
            return out
        small = [preprocess(im, args["max_h"], args["max_w"], max_up=1.0) for im in images]
        redo = [i for i, (a, b) in enumerate(zip(big, small)) if a.shape != b.shape]
        if not redo:
            return out
        cap = {i: None if looped[i] else 2 * lengths[i] + 20 for i in redo}
        # Up to one token past the longest cap, so a reading that never ends counts as over its cap.
        max_len = None if None in cap.values() else max(cap.values()) + 2
        alts, alt_lengths, alt_looped = read([small[i] for i in redo], max_len=max_len)
        for i, alt, n, alt_loop in zip(redo, alts, alt_lengths, alt_looped):
            if alt == out[i] or alt_loop > looped[i] or (cap[i] is not None and n > cap[i]):
                continue
            if alt_loop < looped[i] or mean_logp(small[i], alt) > mean_logp(big[i], out[i]):
                out[i] = alt
        return out

    def choose(images, options):
        """-> for each image, the one of the given readings the model finds likeliest (e.g. "T" or "F")."""
        return [max(options, key=lambda o: mean_logp(x, o)) for x in
                (preprocess(im, args["max_h"], args["max_w"]) for im in images)]

    predict.choose = choose
    return predict

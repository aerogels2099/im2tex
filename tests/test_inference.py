import json

import torch
from PIL import Image, ImageDraw

from im2tex import inference
from im2tex.model import Im2Tex, save_checkpoint
from im2tex.tokenizer import train


def test_load_image_turns_photos_upright_and_puts_transparency_on_white(tmp_path):
    exif = Image.Exif()
    exif[0x0112] = 6  # stored sideways: viewers rotate it 90 degrees
    Image.new("RGB", (40, 20), "white").save(tmp_path / "photo.jpg", exif=exif)
    assert inference.load_image(tmp_path / "photo.jpg").size == (20, 40)
    Image.new("RGBA", (10, 10), (0, 0, 0, 0)).save(tmp_path / "clear.png")
    im = inference.load_image(tmp_path / "clear.png")
    assert im.mode == "RGB" and im.getpixel((0, 0)) == (255, 255, 255)


def test_predictor_loads_a_run_and_returns_one_string_per_image(tmp_path, monkeypatch):
    monkeypatch.setattr(inference, "DATA", tmp_path)
    tok = train([r"x ^ { 2 }", r"\frac { a } { b }"] * 5, vocab_size=120)
    tok.save(str(tmp_path / "tokenizer.json"))
    run = tmp_path / "runs" / "tiny"
    run.mkdir(parents=True)
    (run / "args.json").write_text(json.dumps({"max_h": 64, "max_w": 256}))
    model = Im2Tex(tok.get_vocab_size(), d_model=32, n_layers=1, n_heads=2, pretrained=False, max_tokens=12)
    save_checkpoint(model, run / "best.pt")
    images = []
    for w in (120, 200):
        im = Image.new("RGB", (w, 40), "white")
        ImageDraw.Draw(im).line((10, 20, w - 10, 20), fill="black", width=3)
        images.append(im)
    for beams in (1, 3):
        out = inference.load_predictor("tiny", device="cpu", beams=beams)(images)
        assert len(out) == 2 and all(isinstance(s, str) for s in out)


def test_both_sizes_keeps_the_surer_reading_stops_loops_and_reads_unenlarged_images_once(tmp_path, monkeypatch):
    monkeypatch.setattr(inference, "DATA", tmp_path)
    tok = train([r"x ^ { 2 }", r"\frac { a } { b }"] * 5, vocab_size=120)
    tok.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "runs" / "fake").mkdir(parents=True)
    (tmp_path / "runs" / "fake" / "args.json").write_text(json.dumps({"max_h": 64, "max_w": 256}))
    enlarged, native = (tok.decode(tok.encode(s).ids) for s in (r"x ^ { 2 }", r"\frac { a } { b }"))

    class Model:  # reads x^2 from wide (enlarged) inputs and \frac{a}{b} (or x x x ... if it loops) from narrow ones
        config = {"max_tokens": 12, "vocab_size": tok.get_vocab_size()}

        def __init__(self, sure_of, loops, stops):
            self.sure_of, self.loops, self.stops, self.reads, self.max_len = sure_of, loops, stops, 0, None

        def generate(self, images, bos, eos, max_len=None, stop_loops=False, **kw):
            self.reads += 1
            side = "enlarged" if images.shape[-1] > 128 else "native"
            if side == "native":
                self.max_len = max_len
            if side == self.stops:  # repeated itself: stopped after the first copy, a brace still open
                ids = tok.encode({"enlarged": "x ^ {", "native": r"\frac { a"}[side]).ids
            elif side == "enlarged":
                ids = tok.encode(enlarged).ids
            else:
                ids = [bos] + tok.encode("x").ids[1:2] * (max_len - 1) if self.loops else tok.encode(native).ids
            out = torch.tensor([ids] * len(images))
            return (out, torch.full((len(images),), side == self.stops)) if stop_loops else out

        def __call__(self, images, tokens):  # uniform on the side it's sure of; elsewhere mass on padding
            logits = torch.zeros(1, tokens.shape[1], tok.get_vocab_size())
            if (images.shape[-1] > 128) != (self.sure_of == "enlarged"):
                logits[..., 0] = 10
            return logits

    def predictor(sure_of, loops=False, stops=None):
        model = Model(sure_of, loops, stops)
        monkeypatch.setattr(inference, "load_checkpoint", lambda path, device: (model, None))
        return inference.load_predictor("fake", device="cpu"), model

    def line(w, x0, x1):
        im = Image.new("RGB", (w, 60), "white")
        ImageDraw.Draw(im).line((x0, 30, x1, 30), fill="black", width=3)
        return im

    small, big = line(200, 60, 140), line(400, 20, 380)  # enlarged 2x / already wider than max_w
    predict, model = predictor("native")
    assert predict([big, small]) == [enlarged, native] and model.reads == 2  # the native read is small only
    predict, model = predictor("enlarged")
    assert predict([small]) == [enlarged] and model.reads == 2
    predict, model = predictor("native")
    assert predict([big]) == [enlarged] and model.reads == 1
    predict, model = predictor("native", loops=True)  # sure of a second reading that never ends
    n = len(tok.encode(enlarged).ids) - 2
    assert predict([small]) == [enlarged] and model.max_len <= 2 * n + 22
    # A reading stopped for repeating itself loses to one that wasn't, however sure the model is of it; with
    # the first reading stopped short, the second isn't held to twice its length.
    predict, model = predictor("enlarged", stops="enlarged")
    assert predict([small]) == [native] and model.max_len is None
    predict, model = predictor("native", stops="native")
    assert predict([small]) == [enlarged]
    predict, model = predictor("native", stops="enlarged")  # read once: kept, with its open brace closed
    assert predict([big]) == [tok.decode(tok.encode("x ^ { }").ids)]

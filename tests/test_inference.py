import json

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

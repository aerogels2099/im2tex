import json
import sys
from pathlib import Path

from PIL import Image

from im2tex.normalize import normalize
from im2tex.page import normalize_page

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from label import import_images, load_labels, save_label  # noqa: E402


def _rows(path):
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def test_import_converts_new_images_and_lists_them_unlabeled(tmp_path):
    src, split = tmp_path / "src", tmp_path / "split"
    src.mkdir()
    Image.new("RGBA", (40, 20), (0, 0, 0, 0)).save(src / "slide 1.png")  # transparent background
    Image.new("RGB", (40, 20), "white").save(src / "photo.jpg")
    (src / "notes.txt").write_text("not an image")
    ids = import_images(src, split)
    assert ids == ["photo_jpg", "slide_1_png"]
    assert Image.open(split / "images" / "slide_1_png.png").getpixel((0, 0)) == (255, 255, 255)
    assert [r["image"] for r in _rows(split / "all" / "refs.jsonl")] == [f"../images/{i}.png" for i in ids]
    assert import_images(src, split) == ids  # importing again adds nothing


def test_save_writes_only_valid_labels_and_skips_unusable_images(tmp_path):
    labels = {}
    assert save_label(tmp_path, labels, "a", r"x^{2}") is None
    assert save_label(tmp_path, labels, "b", skip=True) is None
    assert "normalizer" in save_label(tmp_path, labels, "c", r"\frac{a")
    assert "compile" in save_label(tmp_path, labels, "c", r"\notacommand x")
    assert save_label(tmp_path, labels, "a", r"x^{3}") is None  # relabeling replaces
    assert _rows(tmp_path / "refs.jsonl") == [{"id": "a", "image": "images/a.png", "ref": r"x^{3}",
                                               "norm": normalize(r"x^{3}")}]
    assert load_labels(tmp_path) == {"a": {"ref": r"x^{3}"}, "b": {"skip": True}}


def test_page_labels_check_every_formula_and_store_the_normalized_page(tmp_path):
    labels = {}
    page = "Let $x^2$ be\n$$\\frac{a}{b}$$\nand that's all"
    assert save_label(tmp_path, labels, "p", page, pages=True) is None
    assert "never closed" in save_label(tmp_path, labels, "q", "an $x^2 unclosed", pages=True)
    assert r"\notacommand" in save_label(tmp_path, labels, "q", r"ok $x$ but $\notacommand y$", pages=True)
    assert save_label(tmp_path, labels, "r", "only words, no math", pages=True) is None
    rows = _rows(tmp_path / "refs.jsonl")
    assert [r["id"] for r in rows] == ["p", "r"] and rows[0]["norm"] == normalize_page(page)

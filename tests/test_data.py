import io

import numpy as np
import torch
from PIL import Image

from im2tex.augment import (HANDWRITTEN_SYMBOL_PX, PRINTED_SYMBOL_PX, SMALLER_HANDWRITTEN_SYMBOL_PX, augment,
                            draw_scales)
from im2tex.data import FormulaDataset, SizeBatchSampler, collate, fit, ink_box, planned_sizes, preprocess
from im2tex.tokenizer import BOS, LINE, train


def _formula(w=200, h=60):
    img = Image.new("L", (w, h), 255)
    img.paste(0, (50, 20, 120, 35))  # a black bar as "ink"
    return img


def test_ink_box_keeps_a_margin_and_blank_images_stay_whole():
    assert ink_box(_formula()) == (46, 16, 124, 39)
    assert ink_box(Image.new("L", (30, 20), 255)) == (0, 0, 30, 20)


def test_fit_keeps_aspect_ratio_and_limits_enlarging():
    assert fit(2000, 100, max_h=256, max_w=1024) == (1024, 51)
    assert fit(40, 20, max_h=256, max_w=1024) == (80, 40)  # at most 2x larger


def test_preprocess_is_ink_one_paper_zero():
    x = preprocess(_formula(), 256, 1024)
    assert x.shape[0] == 1 and x.min() >= 0 and x.max() <= 1
    assert x[0, 0, 0] == 0 and x.max() == 1


def test_augment_stays_inside_the_planned_size():
    rng = np.random.default_rng(0)
    img = _formula(600, 90)
    for _ in range(200):
        out = augment(img, (400, 60), rng)
        assert out.mode == "L" and out.width <= 400 and out.height <= 60


def test_augment_rarely_erases_a_thin_stroke():
    # A 2 px stroke, like a fraction bar in a small printed render: a max filter (thinning) would wipe it out, and
    # shrinking it for the low-resolution look leaves light gray that a fixed black/white threshold turns blank.
    rng = np.random.default_rng(0)
    img = Image.new("L", (300, 60), 255)
    img.paste(0, (10, 30, 290, 32))
    erased = 0
    for _ in range(2000):
        a = np.asarray(augment(img, (200, 40), rng), dtype=int)
        erased += a.min() > np.median(a) - 25  # nothing clearly darker than the paper
    assert erased < 10


def test_draw_scales_put_median_symbols_between_8_or_12_and_60_px():
    u = draw_scales(np.array(["unimer"] * 20000 + ["mathwriting"] * 20000), np.random.default_rng(0))
    printed, handwritten = u[:20000] * PRINTED_SYMBOL_PX["unimer"], u[20000:] * HANDWRITTEN_SYMBOL_PX
    smallest_old = 0.5 * PRINTED_SYMBOL_PX["unimer"]  # the old 0.5-1.0 range goes a little under 8 px
    assert min(8, smallest_old) - 1e-9 <= printed.min() and printed.max() <= 60 + 1e-9
    assert 12 - 1e-9 <= handwritten.min() and handwritten.max() <= 60 + 1e-9
    assert (printed > 30).mean() > 0.1 and (handwritten < 20).mean() > 0.1  # slide-sized print, small writing
    assert 0.65 < ((u >= 0.5) & (u <= 1.0)).mean() < 0.85  # mostly the old 0.5-1.0
    for source in ("hme100k", "iam", "mixed_lines"):  # photos and long lines come out smaller
        px = draw_scales(np.array([source] * 20000), np.random.default_rng(1)) * SMALLER_HANDWRITTEN_SYMBOL_PX[source]
        assert 12 - 1e-9 <= px.min() and px.max() <= 60 + 1e-9


def test_training_images_fit_inside_the_sizes_their_batches_were_planned_with():
    tok = train([r"x ^ { 2 }", r"\frac { a } { b }"] * 5, vocab_size=120)
    rows, boxes = [], []
    for w, h in [(40, 20), (300, 50), (900, 60), (2000, 120), (60, 400)]:
        img = Image.new("L", (w, h), 255)
        img.paste(0, (w // 4, h // 3, 3 * w // 4, 2 * h // 3))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        rows.append({"image": buf.getvalue(), "label": r"x ^ { 2 }"})
        boxes.append([*ink_box(img), 5])
    index = np.array(boxes)
    sources = ["unimer", "crohme", "synthetic_printed", "mathwriting", "unimer"]
    for scales in [draw_scales(sources, np.random.default_rng(k)) for k in range(10)] + [np.full(5, 2.7),
                                                                                         np.full(5, 0.15)]:
        planned = planned_sizes(index, 256, 1024, scales)
        assert (planned[:, 0] <= 256).all() and (planned[:, 1] <= 1024).all()
        data = FormulaDataset(rows, index, tok, 256, 1024, train=True, scales=scales)
        for i in range(len(rows)):
            for _ in range(5):  # augmentation is random
                x = data[i][0]
                assert x.shape[1] <= planned[i, 0] and x.shape[2] <= planned[i, 1]


def test_lines_start_with_the_line_token_and_formulas_with_bos():
    tok = train([r"x ^ { 2 }", r"\frac { a } { b }"] * 5, vocab_size=120)
    buf = io.BytesIO()
    _formula().save(buf, format="PNG")
    rows = [{"image": buf.getvalue(), "label": "x ^ { 2 }", "source": "mathwriting"},
            {"image": buf.getvalue(), "label": "if $ x ^ { 2 } $ then", "source": "mixed_lines"}]
    data = FormulaDataset(rows, np.array([[*ink_box(_formula()), 5]] * 2), tok, 256, 1024, train=False)
    formula, line = data[0][1], data[1][1]
    assert formula[0] == tok.token_to_id(BOS) and line[0] == tok.token_to_id(LINE)
    assert tok.decode(line.tolist()) == "if $ x ^ { 2 } $ then"


def test_collate_pads_to_multiples_of_16():
    images, tokens = collate([(torch.ones(1, 20, 50), torch.tensor([1, 5, 2])),
                              (torch.ones(1, 33, 17), torch.tensor([1, 2]))])
    assert images.shape == (2, 1, 48, 64)
    assert images[1, 0, 40, 60] == 0 and tokens.tolist() == [[1, 5, 2], [1, 2, 0]]


def test_size_batch_sampler_respects_budgets_and_covers_each_sample_once_per_epoch():
    rng = np.random.default_rng(0)
    n = 3000
    sizes = np.column_stack([rng.integers(16, 256, n), rng.integers(16, 1024, n), rng.integers(3, 300, n)])
    sampler = SizeBatchSampler(sizes, np.arange(n), max_pixels=500_000, max_tokens=4000, max_batch=64, chunk=1000)
    seen = []
    for batch in sampler:
        h, w, t = sizes[batch].max(0)
        assert len(batch) == 1 or (len(batch) * h * w <= 500_000 and len(batch) * t <= 4000 and len(batch) <= 64)
        seen += batch
        if len(seen) >= n:
            break
    assert sorted(seen) == list(range(n))

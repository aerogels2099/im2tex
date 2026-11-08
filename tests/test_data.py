import numpy as np
import torch
from PIL import Image

from im2tex.augment import augment
from im2tex.data import SizeBatchSampler, collate, fit, ink_box, preprocess


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

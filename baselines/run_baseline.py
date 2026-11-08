"""Run a pretrained model over an exported split and write predictions as JSONL ({"id", "pred", "ref"}).

Each baseline lives in its own venv (their pinned dependencies conflict), so this file only needs PIL.
    ~/venvs/pix2tex/bin/python baselines/run_baseline.py pix2tex ~/im2tex-data/im2latex100k/test runs/pix2tex_im2latex100k_test.jsonl
"""
import argparse
import json
from pathlib import Path

from PIL import Image


def load_pix2tex():
    from munch import Munch
    from pix2tex.cli import LatexOCR

    # LatexOCR() with no arguments silently runs on CPU (no_cuda=True); these are its defaults otherwise,
    # except temperature: its default 0.2 samples, so repeated runs disagree. 1e-8 is effectively greedy.
    return LatexOCR(Munch(config="settings/config.yaml", checkpoint="checkpoints/weights.pth",
                          no_cuda=False, no_resize=False, temperature=1e-8))


MODELS = {"pix2tex": load_pix2tex}

ap = argparse.ArgumentParser()
ap.add_argument("model", choices=MODELS)
ap.add_argument("split_dir", type=Path, help="directory with refs.jsonl and images/")
ap.add_argument("out", type=Path)
args = ap.parse_args()

predict = MODELS[args.model]()
args.out.parent.mkdir(parents=True, exist_ok=True)
with (args.split_dir / "refs.jsonl").open(encoding="utf-8") as refs, args.out.open("w", encoding="utf-8") as out:
    for line in refs:
        row = json.loads(line)
        img = Image.open(args.split_dir / "images" / f"{row['id']}.png").convert("RGB")
        try:
            row["pred"] = predict(img)
        except Exception as e:  # a crash on one image is a wrong answer, not a reason to stop
            print(f"{row['id']}: {e!r}")
            row["pred"] = ""
        out.write(json.dumps(row) + "\n")

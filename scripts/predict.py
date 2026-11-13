"""Print the LaTeX for equation images (one equation per image, cropped close) with a trained model.

    python scripts/predict.py screenshot.png photo.jpg [--run run4] [--beams 4] [--cpu] [--one-size] [--line]
"""
import argparse
from pathlib import Path

import torch

from im2tex.gpu import cap_memory
from im2tex.inference import load_image, load_predictor

ap = argparse.ArgumentParser()
ap.add_argument("images", nargs="+", type=Path)
ap.add_argument("--run", default="run4", help="training run whose best checkpoint to use")
ap.add_argument("--beams", type=int, default=1, help="beam search width (1 = greedy)")
ap.add_argument("--cpu", action="store_true", help="run on the CPU, e.g. while the GPU is training")
ap.add_argument("--one-size", action="store_true",
                help="read small images only enlarged, not also at their own size: up to 2x faster, worse on scans")
ap.add_argument("--line", action="store_true",
                help="each image is a line of handwritten words and math: print it as text with $...$ (run4 on)")
args = ap.parse_args()

device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
if device == "cuda":
    cap_memory()  # fail with out-of-memory instead of spilling into system RAM
predict = load_predictor(args.run, device, args.beams, both_sizes=not args.one_size, lines=args.line)
for path in args.images:
    latex = predict([load_image(path)])[0]
    print(latex if len(args.images) == 1 else f"{path}: {latex}")

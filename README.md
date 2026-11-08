# im2tex

Image of a math formula (printed or handwritten) → LaTeX source.

## Setup (WSL2 Ubuntu 24.04, no sudo)

```bash
bash scripts/setup_wsl.sh          # uv, Node, TinyTeX + math packages, ~/venvs/im2tex (torch cu128)
bash baselines/setup_pix2tex.sh    # separate venv: pix2tex pins old dependencies
```

`requirements.lock.txt` records the exact versions known to work (`uv pip freeze`).

Run Python from the repo root with `PYTHONPATH=.` and `~/venvs/im2tex` activated.
Datasets live outside the repo in `$IM2TEX_DATA` (default `~/im2tex-data`), on WSL's own filesystem
because reading from `/mnt/c` is several times slower.

## Evaluation harness

```bash
python -m pytest -q tests
python scripts/export_im2latex.py --split test --n 1000
~/venvs/pix2tex/bin/python baselines/run_baseline.py pix2tex ~/im2tex-data/im2latex100k/test runs/pix2tex.jsonl
python scripts/evaluate.py runs/pix2tex.jsonl      # writes runs/pix2tex.metrics.json
python scripts/benchmark_throughput.py             # training samples/s on this GPU
```

Every model writes the same JSONL (`{"id", "pred", "ref"}`), so `evaluate.py` scores them all the same way.

| Metric | Meaning |
|---|---|
| `bleu4`, `edit_dist` | token BLEU-4 and normalized token edit distance (comparable to im2latex papers) |
| `exprate`, `exprate_le1`, `exprate_le2` | whole formula exactly right / within 1 / within 2 token errors (CROHME style) |
| `render_rate` | prediction compiles with pdflatex |
| `image_match` | prediction renders pixel-identical to the reference (after trimming whitespace) |

Samples whose *reference* fails to compile are excluded from the image metrics.

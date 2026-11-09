# im2tex

Image of a math formula (printed or handwritten) → LaTeX source.

```bash
python scripts/predict.py screenshot.png photo.jpg      # one equation per image, cropped close
```

## Setup (WSL2 Ubuntu 24.04, no sudo)

```bash
bash scripts/setup_wsl.sh          # uv, TinyTeX + math packages, ~/venvs/im2tex (torch cu128)
bash baselines/setup_pix2tex.sh    # separate venv: pix2tex pins old dependencies
```

`requirements.lock.txt` records the exact versions known to work (`uv pip freeze`).

Run Python from the repo root with `PYTHONPATH=.` and `~/venvs/im2tex` activated.
Datasets live outside the repo in `$IM2TEX_DATA` (default `~/im2tex-data`), on WSL's own filesystem
because reading from `/mnt/c` is several times slower.

## Data

```bash
bash scripts/download_data.sh          # UniMER-1M + test, MathWriting, CROHME; HME100K is a manual download
python scripts/prepare_unimer.py       # each split -> refs.jsonl ({"id", "image", "ref", "norm"}) + images/
python scripts/prepare_mathwriting.py  # render pen strokes to PNG
python scripts/prepare_crohme.py       # invert to dark-on-white PNG
python scripts/prepare_crohme2023.py   # test only: handwriting scanned from paper
python scripts/prepare_omnidocbench.py # test only: display equations cut from real slides, textbooks, papers
python scripts/render_synthetic.py ~/im2tex-data/unimer/train/refs.jsonl --n 50000   # 9 math fonts
python scripts/train_tokenizer.py <train refs.jsonl ...> --report <test refs.jsonl ...>
python scripts/build_shards.py         # train/val -> Arrow shards (~6 GB, 14 files)
```

| Split | Rows | Contents |
|---|---|---|
| train | 1,670,906 | UniMER-1M 981k, MathWriting 230k + 396k synthetic + 6k symbols, synthetic printed 49k, CROHME 9k |
| val | 21,560 | MathWriting valid, 0.5% of UniMER and 2% of synthetic printed (held out by id hash) |
| test | as prepared | UniMER-Test spe/cpe/sce/hwe, CROHME 2014/2016/2019/2023, MathWriting test, im2latex-100k, OmniDocBench slides+notes / textbooks / papers+exams |

Labels from every dataset go through `im2tex/normalize.py`, which rewrites LaTeX into one canonical
spelling (`{\bf C}` → `\mathbf { C }`, `x^2_i` → `x _ { i } ^ { 2 }`, `\le` → `\leq`, ...) so formulas that
render the same are written the same. Every rule must leave the rendered image unchanged:

```bash
python scripts/check_normalizer.py <refs.jsonl or formulas.txt>   # errors, idempotency, render unchanged
```

## Training

```bash
python scripts/index_shards.py                       # ink boxes + token counts, once after build_shards
python scripts/train.py --run base --compile         # ConvNeXt-T encoder + 6-layer decoder, ~7 h on an RTX 5060
```

Resumable: rerunning the same `--run` continues from `runs/<run>/last.pt` (saved every 1000 steps).
Logs go to `runs/<run>/log.jsonl`; `best.pt` is the checkpoint with the best validation exact match.

## Labeling your own images

```bash
python scripts/label.py <folder of images> ~/im2tex-data/real/<name>   # then open http://localhost:8765
```

Shows each image beside an editable label and its pdflatex rendering; the saved labels form a test split.

## Evaluation harness

```bash
python -m pytest -q tests
python scripts/export_im2latex.py --split test --n 1000
~/venvs/pix2tex/bin/python baselines/run_baseline.py pix2tex ~/im2tex-data/im2latex100k/test runs/pix2tex.jsonl
python scripts/evaluate.py runs/pix2tex.jsonl      # writes runs/pix2tex.metrics.json
python scripts/benchmark_throughput.py             # training samples/s on this GPU
bash baselines/run_suite.sh im2tex-base ~/venvs/im2tex/bin/python   # one model on every test set, then the table
```

Every model writes the same JSONL (`{"id", "pred", "ref"}`), so `evaluate.py` scores them all the same way.

| Metric | Meaning |
|---|---|
| `bleu4`, `edit_dist` | token BLEU-4 and normalized token edit distance (comparable to im2latex papers) |
| `exprate`, `exprate_le1`, `exprate_le2` | whole formula exactly right / within 1 / within 2 token errors (CROHME style) |
| `render_rate` | prediction compiles with pdflatex |
| `image_match` | prediction renders pixel-identical to the reference (after trimming whitespace) |

Samples whose *reference* fails to compile are excluded from the image metrics.

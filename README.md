# im2tex

Image of a math formula (printed or handwritten) → LaTeX source, or a handwritten line of words and math → text
with `$...$`.

```bash
python scripts/predict.py screenshot.png photo.jpg      # one equation per image, cropped close
python scripts/predict.py --line line.jpg               # one line of handwriting per image
python scripts/read_pdf.py notes.pdf --pages 3 5-7      # handwritten PDF pages with vector ink (OneNote's export)
python scripts/read_pdf.py notes.pdf --markdown notes/  # each note as a Markdown file
```

`read_pdf.py` finds each line of writing from the pen strokes, in the order they were written (a line squeezed
in later above others comes before them), and prints the page as text with `$...$`, a line that is all math as
`$$...$$`. Typed text and pasted pictures are not read. With `--markdown`, each note (OneNote prints a date on
its first page) becomes a `.md` file: its typed title as the heading with the date under it, its pages joined so
that a line cut by a page break is read whole, and printouts or other pictures saved in `pictures/` and shown
where they sit. Formulas are displayed centered, the steps of one solution in one block aligned at their `=`.

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
bash scripts/download_data.sh          # UniMER-1M + test, MathWriting, CROHME, HME100K, IAM, GNHK
python scripts/prepare_unimer.py       # each split -> refs.jsonl ({"id", "image", "ref", "norm"}) + images/
python scripts/prepare_mathwriting.py  # render pen strokes to PNG
python scripts/prepare_crohme.py       # invert to dark-on-white PNG
python scripts/prepare_crohme2023.py   # test only: handwriting scanned from paper
python scripts/prepare_hme100k.py      # train only: photos of handwritten school math
python scripts/prepare_iam.py          # handwritten English text lines, labels written back as written
python scripts/prepare_gnhk.py         # photographed handwriting from many writers, cut into straightened lines
python scripts/prepare_omnidocbench.py # test only: display equations cut from real slides, textbooks, papers
python scripts/render_synthetic.py ~/im2tex-data/unimer/train/refs.jsonl --n 50000   # 9 math fonts
python scripts/make_mixed_lines.py --split train --n 150000   # IAM lines cut at word gaps, formulas glued in
python scripts/make_mixed_lines.py --split val --n 1000
python scripts/prepare_ink_pages.py notes.pdf transcriptions/ ~/im2tex-data/ink_pages  # optional: your own pages
python scripts/train_tokenizer.py <train refs.jsonl ...> --report <test refs.jsonl ...>
python scripts/build_shards.py         # train/val -> Arrow shards (~11 GB, 24 files)
```

| Split | Rows | Contents |
|---|---|---|
| train | 1,906,519 | formulas: UniMER-1M 981k, MathWriting 230k + 396k synthetic + 6k symbols, HME100K 74k, synthetic printed 49k, CROHME 9k; lines: mixed 150k, IAM 6k, GNHK 5k |
| val | 24,195 | formulas: MathWriting valid, 0.5% of UniMER and 2% of synthetic printed (held out by id hash); lines: IAM validation, 1k mixed, GNHK (10% of its training pages) |
| test | as prepared | UniMER-Test spe/cpe/sce/hwe, CROHME 2014/2016/2019/2023, MathWriting test, im2latex-100k, OmniDocBench slides+notes / textbooks / papers+exams; IAM and GNHK test lines |

Lines (IAM's, GNHK's and the mixed ones) are labeled as text with `$...$` math and read from a `<line>` start token
instead of `<bos>`, so the same model reads a formula crop or a line of handwritten notes.

To teach it your own handwriting: transcribe some pages of a vector-ink PDF in the page format (`im2tex/page.py`),
one file per page named `...p<page number>.txt`. `prepare_ink_pages.py` finds their lines, matches them to the
transcriptions and writes the `ink_pages` split, which `build_shards.py` picks up; repeat it in training with
`--repeat ink_pages=50`.

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
python scripts/train.py --run run2 --init ~/im2tex-data/runs/base/best.pt --steps 50000 --lr 1e-4 --warmup 1000 --compile
python scripts/train.py --run run3 --init ~/im2tex-data/runs/run2/best.pt --steps 50000 --lr 1e-4 --warmup 1000 --compile
python scripts/train.py --run run4 --init ~/im2tex-data/runs/run3/best.pt --steps 50000 --lr 1e-4 --warmup 1000 \
    --repeat iam=30 mixed_lines=2 --compile                # IAM lines shown 30 times per pass, mixed lines twice
python scripts/train.py --run run5 --init ~/im2tex-data/runs/run4/best.pt --steps 50000 --lr 1e-4 --warmup 1000 \
    --repeat iam=30 mixed_lines=2 gnhk=30 --compile
python scripts/train.py --run run6 --init ~/im2tex-data/runs/run5/best.pt --steps 50000 --lr 1e-4 --warmup 1000 \
    --repeat iam=30 mixed_lines=2 gnhk=30 ink_pages=50 --compile   # ink_pages: your own pages, if you made them
python scripts/train.py --run run7 --init ~/im2tex-data/runs/run6/best.pt --steps 50000 --lr 1e-4 --warmup 1000 \
    --repeat iam=30 mixed_lines=2 gnhk=30 ink_pages=50 --compile
```

`predict.py` uses `run7` by default. `run2` to `run7` each train the previous run's best weights 50k more
steps at a lower learning rate (~3.5 h each). HME100K was added to the shards before `run3`, the lines (IAM and
mixed) before `run4`, which grows the vocabulary by the `<line>` token, GNHK's lines before `run5`, your own
pages before `run6`, and more of them before `run7`.

Resumable: rerunning the same `--run` continues from `runs/<run>/last.pt` (saved every 1000 steps).
Logs go to `runs/<run>/log.jsonl`; `best.pt` is the checkpoint with the best validation exact match on formulas,
minus the character error rate on lines (`val_cer`) when the shards have lines.
On a Windows laptop, start long jobs through `powershell -ExecutionPolicy Bypass -File scripts\run_awake.ps1 <wsl script>`
so Windows doesn't sleep partway through; the display still turns off as usual.

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
python baselines/run_baseline.py im2tex-run7 ~/im2tex-data/iam/test runs/lines/iam_test.jsonl --line
python scripts/evaluate_pages.py runs/lines/iam_test.jsonl  # lines: character error rate, formulas found
```

Every model writes the same JSONL (`{"id", "pred", "ref"}`), so `evaluate.py` scores them all the same way.

| Metric | Meaning |
|---|---|
| `bleu4`, `edit_dist` | token BLEU-4 and normalized token edit distance (comparable to im2latex papers) |
| `exprate`, `exprate_le1`, `exprate_le2` | whole formula exactly right / within 1 / within 2 token errors (CROHME style) |
| `render_rate` | prediction compiles with pdflatex |
| `image_match` | prediction renders pixel-identical to the reference (after trimming whitespace) |

Samples whose *reference* fails to compile are excluded from the image metrics.

## Licenses

The code is MIT-licensed (see `LICENSE`). No data or model weights are included: the scripts download them,
and each keeps its own license.

| Data | License | Used for |
|---|---|---|
| UniMER-1M, UniMER-Test | Apache-2.0 | training, test |
| MathWriting | CC BY-NC-SA 4.0 (non-commercial) | training, test |
| CROHME 2014–2019 (CoMER's copy) | CC BY-NC-SA 3.0 (non-commercial) | training, test |
| CROHME 2023 | CC BY-NC-SA 3.0 (non-commercial) | test |
| OmniDocBench | research only, non-commercial | test |
| im2latex-100k | see the dataset card (`yuntian-deng/im2latex-100k`) | test |
| HME100K (unofficial Hugging Face copy) | TAL Education's terms, research only | training |
| IAM text lines (Teklia's Hugging Face copy) | IAM's terms, non-commercial research only | training, test |
| GNHK (GoodNotes; a page-level Hugging Face copy) | CC BY 4.0 | training, test |
| Handwritten notes pages (HumynLabs) | unclear: listed as CC BY 4.0, but the pages credit pkalika.in | test |

**Model weights** trained here learn from MathWriting, CROHME, HME100K and IAM, so share them for research and
personal use only, under the same non-commercial terms, and list the datasets above alongside them.
The comparison models in `baselines/` are downloaded at run time and keep their own licenses.

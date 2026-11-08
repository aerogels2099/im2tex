#!/usr/bin/env bash
# Separate venv for pix2tex: it pins old timm / x-transformers versions that conflict with the main env.
#   bash baselines/setup_pix2tex.sh
set -euo pipefail
VENV=$HOME/venvs/pix2tex
[ -x "$VENV/bin/python" ] || uv venv --python /usr/bin/python3.12 "$VENV"
uv pip install --python "$VENV/bin/python" torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "$VENV/bin/python" pix2tex

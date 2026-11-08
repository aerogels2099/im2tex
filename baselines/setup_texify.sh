#!/usr/bin/env bash
# Separate venv for Texify (Donut-style formula OCR); its package pins older transformers.
#   bash baselines/setup_texify.sh
set -euo pipefail
export UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-300}
VENV=$HOME/venvs/texify
[ -x "$VENV/bin/python" ] || uv venv --python 3.12 --managed-python "$VENV"
uv pip install --python "$VENV/bin/python" torch torchvision --index-url https://download.pytorch.org/whl/cu128
# texify 0.2.1 doesn't cap transformers but only works in a window: it needs slow_image_processor_class
# (added ~4.42) and breaks on later config handling
uv pip install --python "$VENV/bin/python" texify "transformers==4.44.2"

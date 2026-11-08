#!/usr/bin/env bash
# Separate venv for UniMERNet + base weights. 0.2.x matches the weights now on Hugging Face (0.1.x
# builds a different attention size), has no torch cap and pins transformers 4.42.4.
#   bash baselines/setup_unimernet.sh
set -euo pipefail
export UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-300}
VENV=$HOME/venvs/unimernet
WEIGHTS=${IM2TEX_DATA:-$HOME/im2tex-data}/models/unimernet_base
[ -x "$VENV/bin/python" ] || uv venv --python 3.12 --managed-python "$VENV"
uv pip install --python "$VENV/bin/python" torch torchvision --index-url https://download.pytorch.org/whl/cu128
# datasets>=3: the old datasets its evaluate dependency would otherwise resolve to breaks on current pyarrow;
# pyarrow<19: its dependencies keep numpy 1.x, and newer pyarrow requires numpy 2
uv pip install --python "$VENV/bin/python" "unimernet==0.2.3" huggingface_hub "datasets>=3" "pyarrow<19"
"$VENV/bin/python" -c "
from huggingface_hub import snapshot_download
snapshot_download('wanderkid/unimernet_base', local_dir='$WEIGHTS')"
"$VENV/bin/python" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

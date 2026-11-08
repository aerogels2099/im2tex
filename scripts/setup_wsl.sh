#!/usr/bin/env bash
# Rootless environment setup for WSL2 Ubuntu 24.04 (no sudo needed).
# Re-runnable: skips anything already installed.
#   bash scripts/setup_wsl.sh
set -euo pipefail

UV_VERSION=0.13.7
TORCH_INDEX=https://download.pytorch.org/whl/cu128   # RTX 50-series (sm_120) needs CUDA >= 12.8
VENV=$HOME/venvs/im2tex
REPO=$(cd "$(dirname "$0")/.." && pwd)
# TeX Live package names: bm is in "tools", mathrsfs in "jknapltx" (+ "rsfs" fonts), upgreek in "was".
TEX_PACKAGES="preview tools jknapltx rsfs was mathtools cancel stmaryrd esint
              fontspec unicode-math lm-math stix2-otf tex-gyre tex-gyre-math xits"

mkdir -p "$HOME/.local/bin"
export PATH=$HOME/.local/bin:$PATH
export UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-300}  # the CUDA wheels are hundreds of MB
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT

echo "== uv $UV_VERSION"
if ! uv --version 2>/dev/null | grep -q "$UV_VERSION"; then
  curl -fsSL "https://github.com/astral-sh/uv/releases/download/$UV_VERSION/uv-x86_64-unknown-linux-gnu.tar.gz" | tar -xz -C "$tmp"
  install -m 755 "$tmp"/uv-x86_64-unknown-linux-gnu/{uv,uvx} "$HOME/.local/bin/"
fi

echo "== TinyTeX"
if ! command -v xelatex >/dev/null; then
  curl -fsSL https://yihui.org/tinytex/install-bin-unix.sh -o "$tmp/install-tinytex.sh"
  sh "$tmp/install-tinytex.sh"
fi
tlmgr update --self
# shellcheck disable=SC2086
tlmgr install $TEX_PACKAGES
# Cambria Math (what PowerPoint/Word equations use) for synthetic rendering; local use only, not redistributed.
if [ -f /mnt/c/Windows/Fonts/cambria.ttc ]; then
  mkdir -p "$HOME/.fonts" && cp -n /mnt/c/Windows/Fonts/cambria.ttc "$HOME/.fonts/" && fc-cache -f "$HOME/.fonts"
fi

echo "== python venv ($VENV)"
# uv's managed Python ships the C headers that torch.compile/Triton need; the system one doesn't
# without the python3.12-dev package (which needs sudo).
[ -x "$VENV/bin/python" ] || uv venv --python 3.12 --managed-python "$VENV"
uv pip install --python "$VENV/bin/python" torch torchvision --index-url "$TORCH_INDEX"
uv pip install --python "$VENV/bin/python" -r "$REPO/requirements.txt"

echo "== versions"
"$VENV/bin/python" - <<'EOF'
import torch
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| available:", torch.cuda.is_available())
x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
print("gpu:", torch.cuda.get_device_name(0), "| matmul ok:", torch.isfinite((x @ x).float()).all().item())
EOF
xelatex --version | head -1
pdflatex --version | head -1

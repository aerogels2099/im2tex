#!/usr/bin/env bash
# Rootless environment setup for WSL2 Ubuntu 24.04 (no sudo needed).
# Re-runnable: skips anything already installed.
#   bash scripts/setup_wsl.sh
set -euo pipefail

UV_VERSION=0.13.7
NODE_VERSION=24.22.2
NODE_SHA256=48cc0e21e048790e54c0cf5749c2725005313d652651c40b7a21feb9697aaf75
TORCH_INDEX=https://download.pytorch.org/whl/cu128   # RTX 50-series (sm_120) needs CUDA >= 12.8
VENV=$HOME/venvs/im2tex
REPO=$(cd "$(dirname "$0")/.." && pwd)
# TeX Live package names: bm is in "tools", mathrsfs in "jknapltx" (+ "rsfs" fonts), upgreek in "was".
TEX_PACKAGES="preview tools jknapltx rsfs was mathtools cancel stmaryrd esint
              fontspec unicode-math lm-math stix2-otf tex-gyre-math xits"

mkdir -p "$HOME/.local/bin" "$HOME/.local/opt"
export PATH=$HOME/.local/bin:$PATH
export UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-300}  # the CUDA wheels are hundreds of MB
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT

echo "== uv $UV_VERSION"
if ! uv --version 2>/dev/null | grep -q "$UV_VERSION"; then
  curl -fsSL "https://github.com/astral-sh/uv/releases/download/$UV_VERSION/uv-x86_64-unknown-linux-gnu.tar.gz" | tar -xz -C "$tmp"
  install -m 755 "$tmp"/uv-x86_64-unknown-linux-gnu/{uv,uvx} "$HOME/.local/bin/"
fi

echo "== node $NODE_VERSION"
NODE_DIR=$HOME/.local/opt/node-v$NODE_VERSION-linux-x64
if [ ! -x "$NODE_DIR/bin/node" ]; then
  f=node-v$NODE_VERSION-linux-x64.tar.xz
  curl -fsSL "https://nodejs.org/dist/v$NODE_VERSION/$f" -o "$tmp/$f"
  echo "$NODE_SHA256  $tmp/$f" | sha256sum -c -
  tar -xJf "$tmp/$f" -C "$HOME/.local/opt"
fi
ln -sf "$NODE_DIR"/bin/{node,npm,npx} "$HOME/.local/bin/"

echo "== TinyTeX"
if ! command -v xelatex >/dev/null; then
  curl -fsSL https://yihui.org/tinytex/install-bin-unix.sh -o "$tmp/install-tinytex.sh"
  sh "$tmp/install-tinytex.sh"
fi
tlmgr update --self
# shellcheck disable=SC2086
tlmgr install $TEX_PACKAGES

echo "== python venv ($VENV)"
# uv's managed Python ships the C headers that torch.compile/Triton need; the system one doesn't
# without the python3.12-dev package (which needs sudo).
[ -x "$VENV/bin/python" ] || uv venv --python 3.12 --managed-python "$VENV"
uv pip install --python "$VENV/bin/python" torch torchvision --index-url "$TORCH_INDEX"
uv pip install --python "$VENV/bin/python" -r "$REPO/requirements.txt"

echo "== katex"
(cd "$REPO/im2tex/render" && npm install --no-audit --no-fund)

echo "== versions"
"$VENV/bin/python" - <<'EOF'
import torch
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| available:", torch.cuda.is_available())
x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
print("gpu:", torch.cuda.get_device_name(0), "| matmul ok:", torch.isfinite((x @ x).float()).all().item())
EOF
xelatex --version | head -1
pdflatex --version | head -1
echo "node $(node --version)"

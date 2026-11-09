#!/usr/bin/env bash
# Download the public datasets into $IM2TEX_DATA (default ~/im2tex-data). Re-runnable: skips finished files.
#   bash scripts/download_data.sh
#
# Licenses (check before releasing a model trained on them):
#   UniMER-1M / UniMER-Test  Apache-2.0                https://huggingface.co/datasets/wanderkid/UniMER_Dataset
#   MathWriting              CC BY-NC-SA 4.0 (non-commercial)  https://arxiv.org/abs/2404.10690
#   CROHME (offline)         CC BY-NC-SA 3.0 (non-commercial). The official TC11 server was unreachable
#                            (2026-11), so this uses the pre-rendered copy from the CoMER repo
#                            (train 2014, test 2014/2016/2019): the version HMER papers report ExpRate on.
#   CROHME 2023              CC BY-NC-SA 3.0 (non-commercial)  https://zenodo.org/records/8428035  (test only)
#   OmniDocBench             research only, non-commercial (README "Copyright Statement"); downloaded by
#                            scripts/prepare_omnidocbench.py  https://huggingface.co/datasets/opendatalab/OmniDocBench
#                            (test only)
# Manual download (terms on the provider's site):
#   HME100K   https://ai.100tal.com/dataset  -> put the archive in $IM2TEX_DATA/hme100k/
set -euo pipefail
DATA=${IM2TEX_DATA:-$HOME/im2tex-data}

fetch() {  # url dest_dir
  mkdir -p "$2"
  local f="$2/$(basename "$1")"
  [ -f "$f.done" ] && { echo "have $f"; return; }
  curl -fL --retry 5 -C - -o "$f" "$1"
  touch "$f.done"
}

UNIMER=https://huggingface.co/datasets/wanderkid/UniMER_Dataset/resolve/main
fetch "$UNIMER/UniMER-Test.zip" "$DATA/unimer"
fetch "$UNIMER/UniMER-1M.zip" "$DATA/unimer"
fetch https://storage.googleapis.com/mathwriting_data/mathwriting-2024.tgz "$DATA/mathwriting"
fetch https://raw.githubusercontent.com/Green-Wood/CoMER/master/data.zip "$DATA/crohme"
fetch https://zenodo.org/records/8428035/files/CROHME23.zip "$DATA/crohme2023"
echo "9eb899ddb0e87c80a17f489bf6162c33  $DATA/crohme2023/CROHME23.zip" | md5sum -c --quiet
echo "done"

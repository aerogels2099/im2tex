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
#   HME100K                  TAL Education's terms, research only  https://ai.100tal.com/dataset . Fetched from an
#                            unofficial Hugging Face copy (ConstantHao/HME100K, the same 74,502 / 24,607 split);
#                            the original terms still apply. Train split only: see scripts/prepare_hme100k.py.
#   IAM (text lines)         IAM's terms, non-commercial research only
#                            https://fki.tic.heia-fr.ch/databases/iam-handwriting-database . Fetched from Teklia's
#                            Hugging Face copy (Teklia/IAM-line, lines scaled to 128 px high); its card says MIT, but
#                            the original terms still apply.
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
HME=https://huggingface.co/datasets/ConstantHao/HME100K/resolve/68ea96bd2527aac3799f94803a5f86b92fd9c2d2/data
fetch "$HME/train-00000-of-00002.parquet" "$DATA/hme100k"
fetch "$HME/train-00001-of-00002.parquet" "$DATA/hme100k"
sha256sum -c --quiet <<EOF
6a64bb7c70e7daaeafdd4397e540cd4a161ce10a2391125f877fbe11bd33af5f  $DATA/hme100k/train-00000-of-00002.parquet
b870559bd247cf95bb1d149f88e96470aa873bbf6cfa1ee6c535fb2e8d776576  $DATA/hme100k/train-00001-of-00002.parquet
EOF
IAM=https://huggingface.co/datasets/Teklia/IAM-line/resolve/fbdad97500ce54635c0d1ba306bf535cb40656cf/data
for split in train validation test; do fetch "$IAM/$split.parquet" "$DATA/iam"; done
sha256sum -c --quiet <<EOF
c9a8f3192f1ac13c54bd900b3e9dcfa9351060b852b0a95e4dff1cf7dd713c89  $DATA/iam/train.parquet
2d197f94c94fe27a8279a3c4e79e3f84cff7afdff2cf617110976595962f6fab  $DATA/iam/validation.parquet
589e05a7db328287d380f33ebc566ebba43baf9f879eb7e102b2706a4212f61b  $DATA/iam/test.parquet
EOF
echo "done"

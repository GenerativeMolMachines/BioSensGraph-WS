#!/bin/bash
set -euo pipefail

uv run dvc exp run -S "operator=translation" -S "train.workers=120"
uv run dvc exp save -n translation

export TMPDIR=/tmp/$USER/pbg_tmp
mkdir -p "$TMPDIR"

./tasks/eval_collect_all.sh \
  --config pbg_configs/train.py \
  --entity-path data/partitions \
  --workers 120 \
  --operator translation

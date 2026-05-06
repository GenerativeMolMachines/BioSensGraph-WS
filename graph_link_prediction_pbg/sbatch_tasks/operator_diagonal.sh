#!/bin/bash
set -euo pipefail

uv run dvc exp run -S "operator=diagonal" -S "train.workers=120"
uv run dvc exp save -n diagonal

export TMPDIR=/tmp/$USER/pbg_tmp
mkdir -p "$TMPDIR"

./tasks/eval_collect_all.sh \
  --config pbg_configs/train.py \
  --entity-path data/partitions \
  --workers 120 \
  --operator diagonal

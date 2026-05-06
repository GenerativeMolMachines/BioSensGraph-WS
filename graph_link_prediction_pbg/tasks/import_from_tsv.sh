#!/bin/bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

uv run torchbiggraph_import_from_tsv \
    --lhs-col=0 --rel-col=1 --rhs-col=2 \
    "$PROJECT_ROOT/pbg_configs/import_from_tsv.py" \
    "$PROJECT_ROOT/resources/train.tsv" \
    "$PROJECT_ROOT/resources/test.tsv"
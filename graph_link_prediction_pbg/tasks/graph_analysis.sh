#!/bin/bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

uv run python graph_report.py \
  --input "$PROJECT_ROOT/resources/triples.csv" \
  --output-dir "$PROJECT_ROOT/reports/graph" \
  --kcore-density-target 1e-2 \
  --triangles-mode sampled \
  --triangles-sample-size 20000 \
  --centrality-sample-k 4000 \
  --centrality-top-n 200
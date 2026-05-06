#!/bin/bash
set -euo pipefail

usage() {
    echo "Usage: $0 --config <path> --checkpoint-root <path> --entity-path <path> --workers <n> --operator <name> --comparator <name> [--edge-samples train,test] [--tmpdir <path>]"
}

log() {
    printf '[eval_collect_epochs] %s\n' "$*"
}

CONFIG=""
CHECKPOINT_ROOT=""
ENTITY_PATH=""
WORKERS=""
OPERATOR=""
COMPARATOR=""
EDGE_SAMPLES="train,test"
TMPDIR_OVERRIDE=""

while [ $# -gt 0 ]; do
    case "$1" in
        --config)
            [ $# -ge 2 ] || { usage; exit 1; }
            CONFIG="$2"
            shift 2
            ;;
        --checkpoint-root)
            [ $# -ge 2 ] || { usage; exit 1; }
            CHECKPOINT_ROOT="$2"
            shift 2
            ;;
        --entity-path)
            [ $# -ge 2 ] || { usage; exit 1; }
            ENTITY_PATH="$2"
            shift 2
            ;;
        --workers)
            [ $# -ge 2 ] || { usage; exit 1; }
            WORKERS="$2"
            shift 2
            ;;
        --operator)
            [ $# -ge 2 ] || { usage; exit 1; }
            OPERATOR="$2"
            shift 2
            ;;
        --comparator)
            [ $# -ge 2 ] || { usage; exit 1; }
            COMPARATOR="$2"
            shift 2
            ;;
        --edge-samples)
            [ $# -ge 2 ] || { usage; exit 1; }
            EDGE_SAMPLES="$2"
            shift 2
            ;;
        --tmpdir)
            [ $# -ge 2 ] || { usage; exit 1; }
            TMPDIR_OVERRIDE="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            exit 1
            ;;
    esac
done

if [ -z "$CONFIG" ] || [ -z "$CHECKPOINT_ROOT" ] || [ -z "$ENTITY_PATH" ] || [ -z "$WORKERS" ] || [ -z "$OPERATOR" ] || [ -z "$COMPARATOR" ]; then
    echo "Missing required arguments."
    usage
    exit 1
fi

if [ ! -f "$CONFIG" ]; then
    echo "Config file not found: $CONFIG"
    exit 1
fi

if [ ! -d "$CHECKPOINT_ROOT" ]; then
    echo "Checkpoint root not found: $CHECKPOINT_ROOT"
    exit 1
fi

METRICS_PARSER="./tasks/awk_parse_metrics.sh"
if [ ! -x "$METRICS_PARSER" ]; then
    echo "Metrics parser is not executable: $METRICS_PARSER"
    exit 1
fi

if [ -n "$TMPDIR_OVERRIDE" ]; then
    TMPDIR="$TMPDIR_OVERRIDE"
fi
if [ -z "${TMPDIR:-}" ] || [[ "${TMPDIR}" == /nfs/* ]]; then
    TMPDIR="/tmp/${USER}/pbg_tmp"
fi
mkdir -p "$TMPDIR"
export TMPDIR
log "Using TMPDIR=$TMPDIR"

IFS=',' read -r -a SAMPLES <<< "$EDGE_SAMPLES"

shopt -s nullglob
CHECKPOINT_DIRS=("$CHECKPOINT_ROOT"/epoch_*)
shopt -u nullglob

if [ ${#CHECKPOINT_DIRS[@]} -eq 0 ]; then
    CHECKPOINT_DIRS=("$CHECKPOINT_ROOT")
fi

mapfile -t CHECKPOINT_DIRS < <(printf '%s\n' "${CHECKPOINT_DIRS[@]}" | sort -V)

LOG_DIR="logs/eval_${OPERATOR}_${COMPARATOR}"
METRICS_DIR="metrics/eval_${OPERATOR}_${COMPARATOR}"
SUMMARY_FILE="metrics/eval_${OPERATOR}_${COMPARATOR}_all_epochs.json"

mkdir -p "$LOG_DIR" "$METRICS_DIR"
rm -f "$LOG_DIR"/*.log "$METRICS_DIR"/*.json "$SUMMARY_FILE"

log "Collecting metrics for comparator=$COMPARATOR operator=$OPERATOR"

for checkpoint_dir in "${CHECKPOINT_DIRS[@]}"; do
    checkpoint_name="$(basename "$checkpoint_dir")"
    if [[ "$checkpoint_name" =~ ^epoch_(.+)$ ]]; then
        epoch="${BASH_REMATCH[1]}"
    else
        epoch="latest"
    fi

    for sample in "${SAMPLES[@]}"; do
        edge_path="${ENTITY_PATH}/${sample}_partitioned"
        log_file="${LOG_DIR}/${sample}_epoch_${epoch}.log"
        metrics_file="${METRICS_DIR}/${sample}_epoch_${epoch}.json"

        if [ ! -f "$edge_path" ] && [ ! -d "$edge_path" ]; then
            echo "Edge path not found: $edge_path"
            exit 1
        fi

        eval_cmd=(
            uv run torchbiggraph_eval
            "$CONFIG"
            -p "workers=${WORKERS}"
            -p "checkpoint_path=${checkpoint_dir}"
            -p "edge_paths=${edge_path}"
        )

        log "Evaluating sample=$sample epoch=$epoch checkpoint=$checkpoint_dir"
        if ! "${eval_cmd[@]}" > "$log_file" 2>&1; then
            echo "Eval failed. See log: $log_file"
            exit 1
        fi
        "$METRICS_PARSER" "$log_file" > "$metrics_file"
    done
done

python3 - "$METRICS_DIR" "$SUMMARY_FILE" "$OPERATOR" "$COMPARATOR" <<'PY'
import json
import re
import sys
from pathlib import Path

metrics_dir = Path(sys.argv[1])
summary_file = Path(sys.argv[2])
operator = sys.argv[3]
comparator = sys.argv[4]

items = []
for path in sorted(metrics_dir.glob("*.json")):
    match = re.match(r"^(train|test)_epoch_(.+)\.json$", path.name)
    if not match:
        continue
    sample, epoch_raw = match.groups()
    epoch = int(epoch_raw) if epoch_raw.isdigit() else epoch_raw
    with path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    metrics["sample"] = sample
    metrics["epoch"] = epoch
    items.append(metrics)

items.sort(key=lambda x: (x["sample"], x["epoch"] if isinstance(x["epoch"], int) else 10**9))

summary = {
    "operator": operator,
    "comparator": comparator,
    "results": items,
}

with summary_file.open("w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=True, indent=2)
PY

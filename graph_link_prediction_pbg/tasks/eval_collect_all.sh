#!/bin/bash
set -euo pipefail

usage() {
    echo "Usage: $0 --config <path> --entity-path <path> --workers <n> --operator <name> [--comparators cos,dot,l2,squared_l2] [--checkpoint-root-base results] [--edge-samples train,test] [--tmpdir <path>]"
}

CONFIG=""
ENTITY_PATH=""
WORKERS=""
OPERATOR=""
COMPARATORS="cos,dot,l2,squared_l2"
CHECKPOINT_ROOT_BASE="results"
EDGE_SAMPLES="train,test"
TMPDIR_OVERRIDE=""

while [ $# -gt 0 ]; do
    case "$1" in
        --config)
            [ $# -ge 2 ] || { usage; exit 1; }
            CONFIG="$2"
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
        --comparators)
            [ $# -ge 2 ] || { usage; exit 1; }
            COMPARATORS="$2"
            shift 2
            ;;
        --checkpoint-root-base)
            [ $# -ge 2 ] || { usage; exit 1; }
            CHECKPOINT_ROOT_BASE="$2"
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

if [ -z "$CONFIG" ] || [ -z "$ENTITY_PATH" ] || [ -z "$WORKERS" ] || [ -z "$OPERATOR" ]; then
    echo "Missing required arguments."
    usage
    exit 1
fi

IFS=',' read -r -a COMPARATOR_LIST <<< "$COMPARATORS"

for comparator in "${COMPARATOR_LIST[@]}"; do
    checkpoint_root="${CHECKPOINT_ROOT_BASE}/${comparator}_${OPERATOR}_checkpoint"
    cmd=(
        ./tasks/eval_collect_epochs.sh
        --config "$CONFIG"
        --checkpoint-root "$checkpoint_root"
        --entity-path "$ENTITY_PATH"
        --workers "$WORKERS"
        --operator "$OPERATOR"
        --comparator "$comparator"
        --edge-samples "$EDGE_SAMPLES"
    )
    if [ -n "$TMPDIR_OVERRIDE" ]; then
        cmd+=(--tmpdir "$TMPDIR_OVERRIDE")
    fi
    "${cmd[@]}"
done


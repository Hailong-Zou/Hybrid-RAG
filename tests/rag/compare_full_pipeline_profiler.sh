#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DEVICE=${PROFILER_DEVICE:-root@192.168.31.189}
DEVICE_PORT=${PROFILER_DEVICE_PORT:-2222}
ROOT=/data/local/tmp/llama-rag-8gen5-kv-layer

tmp_dir=$(mktemp -d)
cleanup() { rm -rf "$tmp_dir"; }
trap cleanup EXIT

for mode in baseline kv layer full; do
    result=$(ssh -p "$DEVICE_PORT" -o BatchMode=yes "$DEVICE" "cat '$ROOT/LATEST_PROFILER_${mode}'")
    case "$result" in
        "$ROOT"/results/profiler-smoke-*) ;;
        *) echo "latest $mode result is not a smoke run: $result" >&2; exit 1 ;;
    esac
    mkdir -p "$tmp_dir/$mode"
    ssh -p "$DEVICE_PORT" -o BatchMode=yes "$DEVICE" "cat '$result/metrics.json'" > "$tmp_dir/$mode/metrics.json"
    printf '%s\n' "$result" > "$tmp_dir/$mode/result-dir.txt"
done

python3 "$SCRIPT_DIR/full_pipeline_profiler.py" compare \
    --baseline "$tmp_dir/baseline" \
    --kv "$tmp_dir/kv" \
    --layer "$tmp_dir/layer" \
    --full "$tmp_dir/full" \
    --output "$tmp_dir/correctness.json" \
    --text-output "$tmp_dir/correctness.txt"

comparison="$ROOT/results/profiler-smoke-comparison-$(date +%Y%m%d-%H%M%S)"
ssh -p "$DEVICE_PORT" -o BatchMode=yes "$DEVICE" "mkdir -p '$comparison'"
ssh -p "$DEVICE_PORT" -o BatchMode=yes "$DEVICE" "cat > '$comparison/correctness.json'" < "$tmp_dir/correctness.json"
ssh -p "$DEVICE_PORT" -o BatchMode=yes "$DEVICE" "cat > '$comparison/correctness.txt'" < "$tmp_dir/correctness.txt"
for mode in baseline kv layer full; do
    result=$(cat "$tmp_dir/$mode/result-dir.txt")
    printf '%s=%s\n' "$mode" "$result"
done | ssh -p "$DEVICE_PORT" -o BatchMode=yes "$DEVICE" "cat > '$comparison/result-dirs.txt'"
ssh -p "$DEVICE_PORT" -o BatchMode=yes "$DEVICE" "printf '%s\n' '$comparison' > '$ROOT/LATEST_PROFILER_SMOKE_COMPARISON'"

echo "PROFILER_COMPARISON_DIR=$comparison"

#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "usage: $0 baseline|kv|layer|full --max-tokens N" >&2
}

MODE=${1:-}
shift || true
MAX_TOKENS=
while [[ $# -gt 0 ]]; do
    case "$1" in
        --max-tokens)
            MAX_TOKENS=${2:-}
            shift 2
            ;;
        *)
            usage
            exit 2
            ;;
    esac
done

case "$MODE" in
    baseline|kv|layer|full) ;;
    *) usage; exit 2 ;;
esac
[[ "$MAX_TOKENS" =~ ^[1-9][0-9]*$ ]] || { usage; exit 2; }

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/../.." && pwd)
DEVICE=${PROFILER_DEVICE:-root@192.168.31.189}
DEVICE_PORT=${PROFILER_DEVICE_PORT:-2222}
ROOT=/data/local/tmp/llama-rag-8gen5-kv-layer
BUILD=/data/junle/build/llama.cpp-rag-android-hexagon-8gen5-kv-layer/bin
FROZEN_COMMIT=918dc3406fd99de965fba6dafc7951ad714c2f23

cd "$REPO"
COMMIT=$(git rev-parse HEAD)
BRANCH=$(git branch --show-current)
if [[ "$COMMIT" != "$FROZEN_COMMIT" ]]; then
    echo "frozen commit mismatch: expected=$FROZEN_COMMIT actual=$COMMIT" >&2
    exit 1
fi
if ! git diff --cached --quiet; then
    echo "staged changes are not allowed for profiler runs" >&2
    exit 1
fi

unexpected=$(git status --porcelain=v1 --untracked-files=all | awk '
    {
        path = substr($0, 4)

        # Existing profiler/test changes.
        if (path ~ /^tests\/(\.gitignore|rag\/(run_full_pipeline_profiler\.sh|full_pipeline_profiler_device\.sh|full_pipeline_profiler\.py|compare_full_pipeline_profiler\.sh|validate_aux_npu_models\.sh|validate_aux_npu_models_device\.sh|validate_aux_npu_models\.py))$/) next

        # Step9E Q4_K/Q6_K Hexagon prototype changes.
        if (path == "ggml/src/ggml-hexagon/ggml-hexagon.cpp") next
        if (path == "ggml/src/ggml-hexagon/htp/hmx-mm-kernels-tiled.h") next
        if (path == "ggml/src/ggml-hexagon/htp/htp-ops.h") next
        if (path == "ggml/src/ggml-hexagon/htp/matmul-ops.c") next
        if (path == "ggml/src/ggml-hexagon/htp/matmul-ops.h") next
        if (path == "tools/server/server-models.cpp") next

        # Zhenyu KV score-selection alignment + HybridRAG per-sequence adaptation.
        if (path == "src/llama-kv-cache-cluster.cpp") next
        if (path == "src/llama-kv-cache-cluster.h") next

        # Step1 async Flash->DRAM / layer-chunk integration.
        if (path == "common/common.cpp") next
        if (path == "include/llama.h") next
        if (path == "src/llama-context.cpp") next
        if (path == "src/llama-model-loader.cpp") next
        if (path == "src/llama-model-loader.h") next
        if (path == "src/llama-model.cpp") next
        if (path == "src/llama-model.h") next
        if (path == "tools/server/server-context.cpp") next

        print
    }')
if [[ -n "$unexpected" ]]; then
    echo "unexpected worktree changes:" >&2
    echo "$unexpected" >&2
    exit 1
fi

ssh -p "$DEVICE_PORT" -o BatchMode=yes -o ConnectTimeout=10 "$DEVICE" true

run_log=$(mktemp)
tmp_dir=$(mktemp -d)
cleanup() {
    rm -f "$run_log"
    rm -rf "$tmp_dir"
}
trap cleanup EXIT

ssh -p "$DEVICE_PORT" -o BatchMode=yes "$DEVICE" \
    "sh -s -- '$MODE' '$MAX_TOKENS' '$COMMIT' '$BRANCH'" \
    < "$SCRIPT_DIR/full_pipeline_profiler_device.sh" | tee "$run_log"

RESULT=$(sed -n 's/^RESULT_DIR=//p' "$run_log" | tail -1)
if [[ -z "$RESULT" ]]; then
    RESULT=$(ssh -p "$DEVICE_PORT" -o BatchMode=yes "$DEVICE" "cat '$ROOT/LATEST_PROFILER_${MODE}'")
fi
case "$RESULT" in
    "$ROOT"/results/profiler-*) ;;
    *) echo "unsafe result path: $RESULT" >&2; exit 1 ;;
esac

git status --porcelain=v1 --untracked-files=all | \
    ssh -p "$DEVICE_PORT" -o BatchMode=yes "$DEVICE" "cat > '$RESULT/git-status.txt'"

sha256sum \
    "$BUILD/llama-server" \
    "$BUILD/libllama.so" \
    "$BUILD/libllama-common.so" \
    "$BUILD/libllama-server-impl.so" \
    "$BUILD/libggml.so" \
    "$BUILD/libggml-base.so" \
    "$BUILD/libggml-cpu.so" \
    "$BUILD/libggml-hexagon.so" \
    "$BUILD/libmtmd.so" > "$tmp_dir/build-hashes.txt"
ssh -p "$DEVICE_PORT" -o BatchMode=yes "$DEVICE" "cat > '$RESULT/build-hashes.txt'" \
    < "$tmp_dir/build-hashes.txt"

fetch_optional() {
    local name=$1
    ssh -p "$DEVICE_PORT" -o BatchMode=yes "$DEVICE" \
        "test ! -f '$RESULT/$name' || cat '$RESULT/$name'" > "$tmp_dir/$name"
}

for name in response.json request.json models.ini server.log server-formal-start-line.txt timing.txt shutdown.txt memory-before.txt memory-after.txt \
    decode-status-before.txt decode-status-after.txt decode-stat-before.txt decode-stat-after.txt \
    pager-files-after.txt deployment-processes-after.txt; do
    fetch_optional "$name"
done

parse_rc=0
python3 "$SCRIPT_DIR/full_pipeline_profiler.py" summarize \
    --result-dir "$tmp_dir" \
    --mode "$MODE" \
    --max-tokens "$MAX_TOKENS" \
    --output "$tmp_dir/metrics.json" \
    --text-output "$tmp_dir/software-parsed.txt" || parse_rc=$?

ssh -p "$DEVICE_PORT" -o BatchMode=yes "$DEVICE" "cat > '$RESULT/metrics.json'" \
    < "$tmp_dir/metrics.json"
ssh -p "$DEVICE_PORT" -o BatchMode=yes "$DEVICE" "cat > '$RESULT/software-parsed.txt'" \
    < "$tmp_dir/software-parsed.txt"

echo "PROFILER_RESULT_DIR=$RESULT"
echo "PROFILER_RUN_VALID=$((parse_rc == 0 ? 1 : 0))"
exit "$parse_rc"

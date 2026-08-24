#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/../.." && pwd)
DEVICE=${PROFILER_DEVICE:-root@192.168.31.189}
DEVICE_PORT=${PROFILER_DEVICE_PORT:-2222}
ROOT=/data/local/tmp/llama-rag-8gen5-kv-layer
FROZEN_COMMIT=918dc3406fd99de965fba6dafc7951ad714c2f23

cd "$REPO"
[[ "$(git rev-parse HEAD)" == "$FROZEN_COMMIT" ]] || { echo "frozen commit mismatch" >&2; exit 1; }
git diff --cached --quiet || { echo "staged changes are not allowed" >&2; exit 1; }

run_log=$(mktemp)
tmp_dir=$(mktemp -d)
cleanup() { rm -f "$run_log"; rm -rf "$tmp_dir"; }
trap cleanup EXIT

ssh -p "$DEVICE_PORT" -o BatchMode=yes -o ConnectTimeout=10 "$DEVICE" true
ssh -p "$DEVICE_PORT" -o BatchMode=yes "$DEVICE" "sh -s" \
    < "$SCRIPT_DIR/validate_aux_npu_models_device.sh" | tee "$run_log"

RESULT=$(sed -n 's/^RESULT_DIR=//p' "$run_log" | tail -1)
if [[ -z "$RESULT" ]]; then
    RESULT=$(ssh -p "$DEVICE_PORT" -o BatchMode=yes "$DEVICE" "cat '$ROOT/LATEST_AUX_NPU_SMOKE'")
fi
case "$RESULT" in
    "$ROOT"/results/aux-npu-smoke-*) ;;
    *) echo "unsafe result path: $RESULT" >&2; exit 1 ;;
esac

for name in models.ini server.log embedding-response.json reranker-response.json shutdown.txt deployment-processes-after.txt; do
    ssh -p "$DEVICE_PORT" -o BatchMode=yes "$DEVICE" "test ! -f '$RESULT/$name' || cat '$RESULT/$name'" > "$tmp_dir/$name"
done

python3 "$SCRIPT_DIR/validate_aux_npu_models.py" \
    --result-dir "$tmp_dir" --output "$tmp_dir/aux-metrics.json"
ssh -p "$DEVICE_PORT" -o BatchMode=yes "$DEVICE" "cat > '$RESULT/aux-metrics.json'" < "$tmp_dir/aux-metrics.json"
echo "AUX_NPU_RESULT_DIR=$RESULT"

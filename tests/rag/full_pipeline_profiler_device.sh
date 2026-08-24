#!/system/bin/sh
set -u

MODE=${1:-}
MAX_TOKENS=${2:-}
COMMIT_SHA=${3:-unknown}
BRANCH=${4:-unknown}

case "$MODE" in
    baseline|kv|layer|full) ;;
    *) echo "usage: $0 baseline|kv|layer|full MAX_TOKENS COMMIT BRANCH" >&2; exit 2 ;;
esac
case "$MAX_TOKENS" in
    ''|*[!0-9]*) echo "max tokens must be a positive integer" >&2; exit 2 ;;
esac
[ "$MAX_TOKENS" -gt 0 ] || exit 2

ROOT=/data/local/tmp/llama-rag-8gen5-kv-layer
SERVER=$ROOT/bin/llama-server
MODEL=/data/local/tmp/zhenyu/models/Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M.gguf
REFERENCE=$ROOT/reference/formal-4k
PORT=18089
if [ "$MAX_TOKENS" -eq 8 ]; then
    RUN_PREFIX=profiler-smoke
else
    RUN_PREFIX=profiler-full
fi
RUN_ID=$RUN_PREFIX-$MODE-$(date +%Y%m%d-%H%M%S)
OUT=$ROOT/results/$RUN_ID
PAGER=$OUT/pager
CONFIG=$OUT/models.ini
REQUEST=$OUT/request.json
PID=
CHILD_PID=
GRACEFUL_TIMEOUT=0
FORCED_STOP_COUNT=0

mkdir -p "$OUT" "$PAGER"
printf '%s\n' "$OUT" > "$ROOT/LATEST_PROFILER_${MODE}"

timestamp() { date +%Y-%m-%dT%H:%M:%S%z; }
log() { printf '%s %s\n' "$(timestamp)" "$*" | tee -a "$OUT/progress.log"; }

list_owned_processes() {
    dest=$1
    : > "$dest"

    for pid in $(pidof llama-server 2>/dev/null || true); do
        [ -d "/proc/$pid" ] || continue
        exe=$(readlink "/proc/$pid/exe" 2>/dev/null || true)

        case "$exe" in
            "$ROOT"/*)
                cmd=$(tr '\000' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)
                printf 'pid=%s exe=%s cmd=%s\n' "$pid" "$exe" "$cmd" >> "$dest"
                ;;
        esac
    done
}

snapshot_process() {
    tag=$1
    pid=$2
    [ -n "$pid" ] || return 0
    [ -d "/proc/$pid" ] || return 0
    exe=$(readlink "/proc/$pid/exe" 2>/dev/null || true)
    [ "$exe" = "$SERVER" ] || return 0
    cat "/proc/$pid/status" > "$OUT/decode-status-$tag.txt" 2>/dev/null || true
    cat "/proc/$pid/stat" > "$OUT/decode-stat-$tag.txt" 2>/dev/null || true
    tr '\000' ' ' < "/proc/$pid/cmdline" > "$OUT/decode-cmdline-$tag.txt" 2>/dev/null || true
}

find_generation_child() {
    for p in /proc/[0-9]*; do
        pid=${p##*/}
        [ "$pid" = "${PID:-}" ] && continue
        exe=$(readlink "$p/exe" 2>/dev/null || true)
        [ "$exe" = "$SERVER" ] || continue
        cmd=$(tr '\000' ' ' < "$p/cmdline" 2>/dev/null || true)
        case "$cmd" in
            *qwen3-4b*) echo "$pid"; return 0 ;;
        esac
    done
    return 1
}

stop_server() {
    [ -n "${PID:-}" ] || return 0
    if [ -d "/proc/$PID" ]; then
        exe=$(readlink "/proc/$PID/exe" 2>/dev/null || true)
        if [ "$exe" != "$SERVER" ]; then
            log "refuse_stop pid=$PID exe=$exe expected=$SERVER"
            PID=
            return 1
        fi
        kill -TERM "$PID" 2>/dev/null || true
        i=0
        while [ -d "/proc/$PID" ] && [ "$i" -lt 60 ]; do
            sleep 1
            i=$((i + 1))
        done
        if [ -d "/proc/$PID" ]; then
            GRACEFUL_TIMEOUT=1
            snapshot_process shutdown-timeout "$PID"
            list_owned_processes "$OUT/deployment-processes-shutdown-timeout.txt"
            exe=$(readlink "/proc/$PID/exe" 2>/dev/null || true)
            if [ "$exe" = "$SERVER" ]; then
                kill -KILL "$PID" 2>/dev/null || true
                FORCED_STOP_COUNT=$((FORCED_STOP_COUNT + 1))
            fi
        fi
    fi
    PID=

    i=0
    while [ "$i" -lt 15 ]; do
        list_owned_processes "$OUT/deployment-processes-post-shutdown.txt"
        [ ! -s "$OUT/deployment-processes-post-shutdown.txt" ] && break
        sleep 1
        i=$((i + 1))
    done
    if [ -s "$OUT/deployment-processes-post-shutdown.txt" ]; then
        GRACEFUL_TIMEOUT=1
        while IFS= read -r line; do
            owned_pid=$(printf '%s\n' "$line" | sed -n 's/^pid=\([0-9][0-9]*\).*/\1/p')
            [ -n "$owned_pid" ] || continue
            exe=$(readlink "/proc/$owned_pid/exe" 2>/dev/null || true)
            case "$exe" in
                "$ROOT"/*)
                    kill -KILL "$owned_pid" 2>/dev/null || true
                    FORCED_STOP_COUNT=$((FORCED_STOP_COUNT + 1))
                    ;;
            esac
        done < "$OUT/deployment-processes-post-shutdown.txt"
    fi
    {
        printf 'graceful_shutdown_timeout=%d\n' "$GRACEFUL_TIMEOUT"
        printf 'forced_stop_count=%d\n' "$FORCED_STOP_COUNT"
    } > "$OUT/shutdown.txt"
}

fail() {
    log "FAILED $*"
    printf '%s\n' "$*" > "$OUT/FAILED"
    stop_server
    exit 1
}
trap 'stop_server' EXIT INT TERM

list_owned_processes "$OUT/deployment-processes-before.txt"
[ ! -s "$OUT/deployment-processes-before.txt" ] || fail old_deployment_processes_present
curl -fsS --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && fail "port_${PORT}_already_in_use"

cat > "$CONFIG" <<'INI'
[qwen3-4b]
model = /data/local/tmp/zhenyu/models/Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M.gguf
ctx-size = 2048
threads = 4
batch-size = 128
ubatch-size = 128
device = HTP0
n-gpu-layers = 99
no-mmap = true
generation-decode-cpu = true
INI

case "$MODE" in
    baseline)
        cat >> "$CONFIG" <<'INI'
cluster-attn = false
layer-chunks = 0
INI
        ;;
    kv)
        cat >> "$CONFIG" <<'INI'
cluster-attn = true
cluster-k = 4
max-resident-clusters = 6
cluster-update-interval = 1
cluster-topk-tokens = 1024
cluster-max-prefetch = 4
cluster-score-select = true
layer-chunks = 0
INI
        ;;
    layer)
        cat >> "$CONFIG" <<'INI'
cluster-attn = false
layer-chunks = 9
INI
        ;;
    full)
        cat >> "$CONFIG" <<'INI'
cluster-attn = true
cluster-k = 4
max-resident-clusters = 6
cluster-update-interval = 1
cluster-topk-tokens = 1024
cluster-max-prefetch = 4
cluster-score-select = true
layer-chunks = 9
INI
        ;;
esac

cat >> "$CONFIG" <<'INI'

[qwen3-0.6b-base]
model = /data/local/tmp/models/Qwen3-0.6B-Q8_0.gguf
ctx-size = 4096
threads = 4
n-gpu-layers = 0
chat-template = chatml
skip-chat-parsing = true

[qwen3-embedding-0.6b]
model = /data/local/tmp/models/Qwen3-Embedding-0.6B-Q8_0.gguf
ctx-size = 4096
threads = 4
device = HTP0
n-gpu-layers = 99
embeddings = true
verbosity = 5

[qwen3-reranker-0.6b]
model = /data/local/tmp/models/Qwen3-Reranker-0.6B-q8_0.gguf
ctx-size = 4096
threads = 4
device = HTP0
n-gpu-layers = 99
reranking = true
verbosity = 5
INI

cp "$REFERENCE/request.json" "$REQUEST" || fail reference_request_copy_failed
sed -i "s/\"max_tokens\": 8/\"max_tokens\": $MAX_TOKENS/" "$REQUEST"
sed -i "s/\"generation_decode_steps\": 8/\"generation_decode_steps\": $MAX_TOKENS/" "$REQUEST"
sed -i 's/"generation_subquery_decode_steps": 8/"generation_subquery_decode_steps": 64/' "$REQUEST"
sed -i 's/"document_embedding_backend": "cpu"/"document_embedding_backend": "npu"/' "$REQUEST"
sed -i 's/"query_embedding_backend": "cpu"/"query_embedding_backend": "npu"/' "$REQUEST"
sed -i 's/"reranking_backend": "cpu"/"reranking_backend": "npu"/' "$REQUEST"
sed -i 's/"generation_candidate_repeats": 3/"generation_candidate_repeats": 1/' "$REQUEST"
sed -i -E 's/"temperature"[[:space:]]*:[[:space:]]*[0-9.]+/"temperature": 0.1/' "$REQUEST"

{
    printf 'run_id=%s\n' "$RUN_ID"
    printf 'mode=%s\n' "$MODE"
    printf 'commit=%s\n' "$COMMIT_SHA"
    printf 'branch=%s\n' "$BRANCH"
    printf 'candidate_count=4\n'
    printf 'generation_candidate_repeats=1\n'
    printf 'generation_prefill_backend=npu\n'
    printf 'generation_decode_backend=cpu\n'
    printf 'document_embedding_model_device=HTP0\n'
    printf 'query_embedding_model_device=HTP0\n'
    printf 'reranking_model_device=HTP0\n'
    printf 'query_expansion_model_device=CPU\n'
    printf 'max_tokens=%s\n' "$MAX_TOKENS"
    printf 'temperature=0.1\nseed=1234\ncache_prompt=false\n'
    printf 'cluster_attn=%s\n' "$(case "$MODE" in kv|full) echo true ;; *) echo false ;; esac)"
    printf 'layer_chunks=%s\n' "$(case "$MODE" in layer|full) echo 9 ;; *) echo 0 ;; esac)"
    printf 'max_resident_clusters=%s\n' "$(case "$MODE" in kv|full) echo 6 ;; *) echo disabled ;; esac)"
    uname -a
    printf 'boot_id='; cat /proc/sys/kernel/random/boot_id
} > "$OUT/manifest.txt"

cat > "$OUT/hashes.txt" <<'HASHES'
4537c38cdd20d514a1686d2e97622109b64b3acfac3a541acb507a80184f1308  /data/local/tmp/llama-rag-8gen5-kv-layer/bin/llama-server
93143b8a124c3ea5605afd0a7540b1a84d8fb148787e83c8729b26c36cc76643  /data/local/tmp/llama-rag-8gen5-kv-layer/lib/libllama.so
cddb622a8dcdbc2e171cf4d42255504d006c8bb774d8e5d0754d08ceb0ce3783  /data/local/tmp/llama-rag-8gen5-kv-layer/lib/libllama-common.so
edd558b4362649e79b68cb9aef96db041e8411c834bde606596fb3d8f0a19ff3  /data/local/tmp/llama-rag-8gen5-kv-layer/lib/libllama-server-impl.so
c4afcfffe7abca1059723bb9238b5550f3b87eeac0bf7b99ccd103bb8b3af638  /data/local/tmp/llama-rag-8gen5-kv-layer/lib/libggml.so
05dd85042a5b2ec0dc3ce40b6dac317e0358e9a9ef0314876d4feb267c729a08  /data/local/tmp/llama-rag-8gen5-kv-layer/lib/libggml-base.so
260debb02d74c6856f3198e6ff8db3afe1ac362beb1081a641b90d5744395a87  /data/local/tmp/llama-rag-8gen5-kv-layer/lib/libggml-cpu.so
247f519ba02d61431449af4d2f001161f56a3040f274b56728c43c9ea49d97a9  /data/local/tmp/llama-rag-8gen5-kv-layer/lib/libggml-hexagon.so
74fd562a8eb130e1364fefc90d200a800de21872442d41324cf2ceb4b8e27305  /data/local/tmp/llama-rag-8gen5-kv-layer/lib/libmtmd.so
7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5  /data/local/tmp/zhenyu/models/Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M.gguf
HASHES

sha256sum "$CONFIG" "$REQUEST" >> "$OUT/hashes.txt" 2>&1 || fail hash_capture_failed

find "$PAGER" -type f -print 2>/dev/null | sort > "$OUT/pager-files-before.txt" || true
grep -E '^(MemAvailable|SwapFree):' /proc/meminfo > "$OUT/memory-before.txt"

export LD_LIBRARY_PATH="$ROOT/lib"
export ADSP_LIBRARY_PATH="$ROOT/lib"
export DSP_LIBRARY_PATH="$ROOT/lib"
export LLAMA_KV_PAGER_DIR="$PAGER"

case "$MODE" in
    layer|full)
        export LLAMA_LAYER_CHUNKS_ADVICE=pageout
        ;;
    *)
        unset LLAMA_LAYER_CHUNKS_ADVICE
        ;;
esac
log "layer_chunks_advice=${LLAMA_LAYER_CHUNKS_ADVICE:-pageout}"

unset LLAMA_REPACK_PREPACK_DUMP_DIR
export LLAMA_REPACK_PREPACK_LOAD_DIR=/data/local/tmp/llama-rag-8gen5-kv-layer/prepack-q4km-9chunks
export LLAMA_OFFLINE_PREPACK=1
log "offline_prepack=${LLAMA_OFFLINE_PREPACK}"
log "repack_prepack_load_dir=${LLAMA_REPACK_PREPACK_LOAD_DIR}"
log "phase=server_start mode=$MODE"
LLAMA_REPACK_ASYNC_CPU=1 LLAMA_REPACK_ASYNC_CPU=1 "$SERVER" --models-preset "$CONFIG" --models-max 4 --host 127.0.0.1 --port "$PORT" > "$OUT/server.log" 2>&1 &
PID=$!
printf '%s\n' "$PID" > "$OUT/server.pid"
sleep 1
[ -d "/proc/$PID" ] || fail server_exited_early
exe=$(readlink "/proc/$PID/exe" 2>/dev/null || true)
[ "$exe" = "$SERVER" ] || fail "exe_mismatch:$exe"
tr '\000' ' ' < "/proc/$PID/cmdline" > "$OUT/server.proc-cmdline.txt"

ready=0
i=0
while [ "$i" -lt 180 ]; do
    if curl -fsS --max-time 3 "http://127.0.0.1:$PORT/health" > "$OUT/health.json" 2>/dev/null; then
        ready=1
        break
    fi
    [ -d "/proc/$PID" ] || fail server_died_before_health
    sleep 1
    i=$((i + 1))
done
[ "$ready" -eq 1 ] || fail health_timeout
log "phase=ready waited_s=$i"

for model_name in qwen3-4b qwen3-0.6b-base qwen3-embedding-0.6b qwen3-reranker-0.6b; do
    log "phase=model_load_start model=$model_name"
    curl -sS --max-time 900 -H 'Content-Type: application/json' --data "{\"model\":\"$model_name\"}" \
        -o "$OUT/load-$model_name.json" "http://127.0.0.1:$PORT/models/load" || fail "load_request_failed:$model_name"
    grep -q '"success":true' "$OUT/load-$model_name.json" || fail "model_load_failed:$model_name"
    loaded=0
    i=0
    while [ "$i" -lt 600 ]; do
        code=$(curl -sS --max-time 5 -o "$OUT/props-$model_name.json" -w '%{http_code}' \
            "http://127.0.0.1:$PORT/props?model=$model_name&autoload=false" 2>/dev/null || true)
        if [ "$code" = 200 ]; then
            loaded=1
            break
        fi
        [ -d "/proc/$PID" ] || fail "server_died_loading:$model_name"
        sleep 1
        i=$((i + 1))
    done
    [ "$loaded" -eq 1 ] || fail "model_load_timeout:$model_name"
    log "phase=model_load_done model=$model_name waited_s=$i"
done

    # Unified CPU-decode warm-up for all profiler modes.
    CHILD_PID_PREWARM=$(find_generation_child || true)
    [ -n "$CHILD_PID_PREWARM" ] || fail generation_decode_child_not_found_before_warmup
    printf '%s\n' "$CHILD_PID_PREWARM" > "$OUT/decode-child-prewarm.pid"

    WARM_REQUEST="$OUT/warmup-request.json"
    cp "$REQUEST" "$WARM_REQUEST"

    sed -i \
        's/"max_tokens"[[:space:]]*:[[:space:]]*[0-9][0-9]*/"max_tokens": 4/' \
        "$WARM_REQUEST"

    grep -Eq '"max_tokens"[[:space:]]*:[[:space:]]*4([,}])' \
        "$WARM_REQUEST" || fail warmup_request_patch_failed

    log "WARMUP_START timestamp=$(timestamp) mode=$MODE max_tokens=4 pid=$CHILD_PID_PREWARM"

    curl -sS --max-time 1800 \
        -H 'Content-Type: application/json' \
        --data-binary "@$WARM_REQUEST" \
        -o "$OUT/warmup-response.json" \
        -w 'http_code=%{http_code}\ntime_total_s=%{time_total}\nsize_download=%{size_download}\n' \
        "http://127.0.0.1:$PORT/v1/rag" \
        > "$OUT/warmup-timing.txt" \
        2> "$OUT/warmup-curl-stderr.txt"

    warm_rc=$?
    [ "$warm_rc" -eq 0 ] || fail "warmup_curl_rc=$warm_rc"
    grep -q 'http_code=200' "$OUT/warmup-timing.txt" \
        || fail warmup_http_not_200

    CHILD_PID=$(find_generation_child || true)
    [ -n "$CHILD_PID" ] || fail generation_decode_child_not_found_after_warmup

    [ "$CHILD_PID" = "$CHILD_PID_PREWARM" ] \
        || fail "decode_child_changed_during_warmup:${CHILD_PID_PREWARM}->${CHILD_PID}"

    printf '%s\n' "$CHILD_PID" > "$OUT/decode-child.pid"

    find "$PAGER" -type f -printf '%p %s\n' 2>/dev/null \
        | sort > "$OUT/pager-files-after-warmup.txt" || true

    warm_pager_count=$(find "$PAGER" -type f 2>/dev/null \
        | wc -l | tr -d '[:space:]')

    [ "$warm_pager_count" -eq 0 ] \
        || fail "warmup_pager_not_empty:$warm_pager_count"

    sleep 1

    snapshot_process before "$CHILD_PID"

    grep -E '^(MemAvailable|SwapFree):' /proc/meminfo \
        > "$OUT/memory-workload-before.txt"

    rss_kb=$(awk '/^VmRSS:/  {print $2}' "$OUT/decode-status-before.txt")
    anon_kb=$(awk '/^RssAnon:/ {print $2}' "$OUT/decode-status-before.txt")
    file_kb=$(awk '/^RssFile:/ {print $2}' "$OUT/decode-status-before.txt")
    swap_kb=$(awk '/^VmSwap:/  {print $2}' "$OUT/decode-status-before.txt")

    log "WARMUP_END timestamp=$(timestamp) mode=$MODE pid=$CHILD_PID"

SERVER_FORMAL_START_LINE=$(wc -l < "$OUT/server.log" 2>/dev/null || echo 0)
printf '%s\n' "$SERVER_FORMAL_START_LINE" \
    > "$OUT/server-formal-start-line.txt"
log "FORMAL_LOG_START_LINE line=$SERVER_FORMAL_START_LINE"

    log "PRE_WORKLOAD_RESIDENCY pid=$CHILD_PID VmRSS_kB=$rss_kb RssAnon_kB=$anon_kb RssFile_kB=$file_kb VmSwap_kB=$swap_kb"

log "PROFILER_WORKLOAD_START timestamp=$(timestamp) mode=$MODE max_tokens=$MAX_TOKENS"
start_ns=$(date +%s%N 2>/dev/null || date +%s000000000)
curl -sS --max-time 1800 -H 'Content-Type: application/json' --data-binary "@$REQUEST" \
    -o "$OUT/response.json" \
    -w 'http_code=%{http_code}\ntime_total_s=%{time_total}\nsize_download=%{size_download}\n' \
    "http://127.0.0.1:$PORT/v1/rag" > "$OUT/timing.txt" 2> "$OUT/curl-stderr.txt"
rc=$?
end_ns=$(date +%s%N 2>/dev/null || date +%s000000000)
printf 'wall_ns_start=%s\nwall_ns_end=%s\n' "$start_ns" "$end_ns" >> "$OUT/timing.txt"
[ "$rc" -eq 0 ] || fail "curl_rc=$rc"
grep -q 'http_code=200' "$OUT/timing.txt" || fail http_not_200

echo "===== STAGE METRICS ====="
grep -oE '"(embedding|indexing|query_expand|query_embedding|searching|reranking|generation|total)"[[:space:]]*:[[:space:]]*[0-9]+' "$OUT/response.json" || true
echo "===== GENERATION BREAKDOWN ====="
grep -oE '"generation_(prefill|decode)_ms"[[:space:]]*:[[:space:]]*[0-9]+' "$OUT/response.json" || true

log "PROFILER_WORKLOAD_END timestamp=$(timestamp) mode=$MODE max_tokens=$MAX_TOKENS"

# Run 2 on the exact same loaded server/model processes.

CHILD_PID_AFTER=$(find_generation_child || true)

if [ -n "$CHILD_PID_AFTER" ]; then
    if [ "$CHILD_PID_AFTER" != "$CHILD_PID" ]; then
        log "POST_WORKLOAD_DECODE_CHILD_CHANGED before=$CHILD_PID after=$CHILD_PID_AFTER"
    else
        log "POST_WORKLOAD_DECODE_CHILD_SAME pid=$CHILD_PID_AFTER"
    fi

    snapshot_process after "$CHILD_PID_AFTER"
else
    log "POST_WORKLOAD_DECODE_CHILD_GONE before=$CHILD_PID"
    printf 'before_pid=%s\n' "$CHILD_PID" \
        > "$OUT/decode-child-after-missing.txt"
fi

grep -E '^(MemAvailable|SwapFree):' /proc/meminfo > "$OUT/memory-after.txt"
find "$PAGER" -type f -printf '%p %s\n' 2>/dev/null | sort > "$OUT/pager-files-live.txt" || true

stop_server
sleep 3
find "$PAGER" -type f -print 2>/dev/null | sort > "$OUT/pager-files-after.txt" || true
list_owned_processes "$OUT/deployment-processes-after.txt"

grep -E 'GENERATION_DECODE_CONTEXT_CONFIG|HANDOFF_NATIVE_RESTORE_DONE|KV_HANDOFF_REBUILD_(BEGIN|DONE)|KV_CLUSTER_SUMMARY|KV_EVICT_COMPLETE|KV_PAGE_IN_COMPLETE|KV_PAGER_SUMMARY|KV_PAGER_CLEANUP|KV_.*METADATA_ERROR|LAYER_CHUNK_SUMMARY|LAYER_MADVISE_SUMMARY|load_model: loading model|offload|HTP0|fallback|unsupported' \
    "$OUT/server.log" > "$OUT/software_stats.txt" 2>/dev/null || true

touch "$OUT/COMPLETE"
log "suite=complete mode=$MODE result=$OUT pager_after=$(wc -l < "$OUT/pager-files-after.txt")"
printf 'RESULT_DIR=%s\n' "$OUT"

#!/system/bin/sh
set -u

ROOT=/data/local/tmp/llama-rag-8gen5-kv-layer
SERVER=$ROOT/bin/llama-server
PORT=18090
RUN_ID=aux-npu-smoke-$(date +%Y%m%d-%H%M%S)
OUT=$ROOT/results/$RUN_ID
CONFIG=$OUT/models.ini
PID=
GRACEFUL_TIMEOUT=0
FORCED_STOP_COUNT=0

mkdir -p "$OUT"
printf '%s\n' "$OUT" > "$ROOT/LATEST_AUX_NPU_SMOKE"

timestamp() { date +%Y-%m-%dT%H:%M:%S%z; }
log() { printf '%s %s\n' "$(timestamp)" "$*" | tee -a "$OUT/progress.log"; }

list_owned_processes() {
    dest=$1
    : > "$dest"
    for p in /proc/[0-9]*; do
        pid=${p##*/}
        exe=$(readlink "$p/exe" 2>/dev/null || true)
        case "$exe" in
            "$ROOT"/*)
                cmd=$(tr '\000' ' ' < "$p/cmdline" 2>/dev/null || true)
                printf 'pid=%s exe=%s cmd=%s\n' "$pid" "$exe" "$cmd" >> "$dest"
                ;;
        esac
    done
}

stop_server() {
    if [ -n "${PID:-}" ] && [ -d "/proc/$PID" ]; then
        exe=$(readlink "/proc/$PID/exe" 2>/dev/null || true)
        if [ "$exe" = "$SERVER" ]; then
            kill -TERM "$PID" 2>/dev/null || true
            i=0
            while [ -d "/proc/$PID" ] && [ "$i" -lt 60 ]; do
                sleep 1
                i=$((i + 1))
            done
            if [ -d "/proc/$PID" ]; then
                GRACEFUL_TIMEOUT=1
                exe=$(readlink "/proc/$PID/exe" 2>/dev/null || true)
                if [ "$exe" = "$SERVER" ]; then
                    kill -KILL "$PID" 2>/dev/null || true
                    FORCED_STOP_COUNT=$((FORCED_STOP_COUNT + 1))
                fi
            fi
        fi
    fi
    PID=
    sleep 2
    list_owned_processes "$OUT/deployment-processes-after.txt"
    if [ -s "$OUT/deployment-processes-after.txt" ]; then
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
        done < "$OUT/deployment-processes-after.txt"
        sleep 1
        list_owned_processes "$OUT/deployment-processes-after.txt"
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
    printf 'RESULT_DIR=%s\n' "$OUT"
    exit 1
}
trap 'stop_server' EXIT INT TERM

list_owned_processes "$OUT/deployment-processes-before.txt"
[ ! -s "$OUT/deployment-processes-before.txt" ] || fail old_deployment_processes_present
curl -fsS --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && fail "port_${PORT}_already_in_use"

cat > "$CONFIG" <<'INI'
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

cat > "$OUT/embedding-request.json" <<'JSON'
{"model":"qwen3-embedding-0.6b","input":"Hybrid retrieval combines dense and sparse evidence.","encoding_format":"float"}
JSON
cat > "$OUT/reranker-request.json" <<'JSON'
{"model":"qwen3-reranker-0.6b","query":"Which passage discusses hybrid retrieval?","top_n":2,"documents":["Hybrid retrieval combines dense and sparse evidence.","The weather is warm today."]}
JSON

export LD_LIBRARY_PATH="$ROOT/lib"
export ADSP_LIBRARY_PATH="$ROOT/lib"
export DSP_LIBRARY_PATH="$ROOT/lib"

log "phase=server_start"
"$SERVER" --models-preset "$CONFIG" --models-max 2 --host 127.0.0.1 --port "$PORT" > "$OUT/server.log" 2>&1 &
PID=$!
printf '%s\n' "$PID" > "$OUT/server.pid"
sleep 1
[ -d "/proc/$PID" ] || fail server_exited_early
[ "$(readlink "/proc/$PID/exe" 2>/dev/null || true)" = "$SERVER" ] || fail server_exe_mismatch

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

for model_name in qwen3-embedding-0.6b qwen3-reranker-0.6b; do
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

log "phase=embedding_request_start"
curl -sS --max-time 600 -H 'Content-Type: application/json' --data-binary "@$OUT/embedding-request.json" \
    -o "$OUT/embedding-response.json" -w 'http_code=%{http_code}\ntime_total_s=%{time_total}\nsize_download=%{size_download}\n' \
    "http://127.0.0.1:$PORT/v1/embeddings" > "$OUT/embedding-timing.txt" 2> "$OUT/embedding-stderr.txt" || fail embedding_request_failed
grep -q 'http_code=200' "$OUT/embedding-timing.txt" || fail embedding_http_not_200
grep -q '"embedding"' "$OUT/embedding-response.json" || fail embedding_output_missing
log "phase=embedding_request_done"

log "phase=reranker_request_start"
curl -sS --max-time 600 -H 'Content-Type: application/json' --data-binary "@$OUT/reranker-request.json" \
    -o "$OUT/reranker-response.json" -w 'http_code=%{http_code}\ntime_total_s=%{time_total}\nsize_download=%{size_download}\n' \
    "http://127.0.0.1:$PORT/v1/rerank" > "$OUT/reranker-timing.txt" 2> "$OUT/reranker-stderr.txt" || fail reranker_request_failed
grep -q 'http_code=200' "$OUT/reranker-timing.txt" || fail reranker_http_not_200
grep -q '"results"' "$OUT/reranker-response.json" || fail reranker_output_missing
log "phase=reranker_request_done"

stop_server
grep -E 'load_model: loading model|HTP0|offload|fallback|unsupported|fatal|assert|segmentation' "$OUT/server.log" \
    > "$OUT/backend-evidence.txt" 2>/dev/null || true
touch "$OUT/COMPLETE"
log "suite=complete result=$OUT"
printf 'RESULT_DIR=%s\n' "$OUT"
trap - EXIT INT TERM

#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(dirname -- "$script_dir")"
model="${DS4_MODEL:-/home/peppe200175/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf}"
port="${DS4_TRACE_PORT:-18080}"
run_id="${DS4_TRACE_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
run_dir="$repo_dir/logs/expert_trace_10prompts_$run_id"
trace_file="$run_dir/expert_trace.jsonl"
server_log="$run_dir/server.log"
server_stdout="$run_dir/server.stdout.log"
results_file="$run_dir/benchmark_results.json"
analysis_dir="$run_dir/analysis"

if [[ -e "$run_dir" ]]; then
    printf 'Refusing to append to an existing trace run: %s\n' "$run_dir" >&2
    exit 2
fi
if [[ ! -x "$repo_dir/ds4-server" ]]; then
    printf 'Missing executable: %s\n' "$repo_dir/ds4-server" >&2
    exit 2
fi
if [[ ! -r "$model" ]]; then
    printf 'Model is not readable: %s\n' "$model" >&2
    exit 2
fi
if pgrep -x ds4 >/dev/null || pgrep -x ds4-server >/dev/null; then
    printf 'Another ds4 process is active; stop it before collecting an isolated trace.\n' >&2
    pgrep -af 'ds4|ds4-server' >&2 || true
    exit 2
fi

mkdir -p "$run_dir"

env \
    DS4_LOCK_FILE="/tmp/ds4-expert-trace-$run_id.lock" \
    DS4_EXPERT_TRACE="$trace_file" \
    DS4_CUDA_MMQ=0 \
    DS4_CUDA_NO_Q8_F16_CACHE=1 \
    DS4_CUDA_STREAMING_READ_THREADS=4 \
    DS4_CUDA_STREAMING_SMALL_MISS_PARALLEL=1 \
    DS4_CUDA_STREAMING_PERSISTENT_READERS=1 \
    DS4_CUDA_DECODE_CACHE_LRU=1 \
    DS4_CUDA_STREAMING_PREFILL_SHARED_OVERLAP=0 \
    DS4_CUDA_PROMPT_EXPERT_CACHE=0 \
    DS4_CUDA_WEIGHT_ARENA_CHUNK_MB=256 \
    "$repo_dir/ds4-server" \
        --cuda \
        -m "$model" \
        --ssd-streaming \
        --ssd-streaming-cache-experts 10GB \
        --prefill-chunk 128 \
        --ctx 4096 \
        --tokens 32 \
        --host 127.0.0.1 \
        --port "$port" \
        >"$server_stdout" 2>"$server_log" &
server_pid=$!

cleanup() {
    if kill -0 "$server_pid" 2>/dev/null; then
        kill "$server_pid"
        wait "$server_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

ready=0
for _ in $(seq 1 120); do
    if ! kill -0 "$server_pid" 2>/dev/null; then
        printf 'Trace server exited during startup. See %s\n' "$server_log" >&2
        exit 1
    fi
    if python3 -c \
        "import urllib.request; urllib.request.urlopen('http://127.0.0.1:${port}/v1/models', timeout=2).read()" \
        >/dev/null 2>&1
    then
        ready=1
        break
    fi
    sleep 1
done

if [[ "$ready" != 1 ]]; then
    printf 'Trace server did not become ready. See %s\n' "$server_log" >&2
    exit 1
fi

python3 "$script_dir/bench_10_prompts.py" \
    --url "http://127.0.0.1:${port}" \
    --max-tokens 32 \
    --output "$results_file"

cleanup
trap - EXIT INT TERM

python3 "$script_dir/analyze_expert_trace.py" \
    "$trace_file" \
    --benchmark-results "$results_file" \
    --decode-reclaim-gib 3.38 \
    --output-dir "$analysis_dir"

printf 'Trace: %s\n' "$trace_file"
printf 'Analysis: %s\n' "$analysis_dir"

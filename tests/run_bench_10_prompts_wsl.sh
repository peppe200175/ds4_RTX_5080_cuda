#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(dirname -- "$script_dir")"
model="${DS4_MODEL:-/home/peppe200175/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf}"
port="${DS4_BENCH_PORT:-18080}"
max_tokens="${DS4_BENCH_MAX_TOKENS:-32}"
server_log="$repo_dir/logs/bench_10_prompts_wsl_optimized_server.log"
server_stdout="$repo_dir/logs/bench_10_prompts_wsl_optimized_server.stdout.log"
result_file="$repo_dir/logs/bench_10_prompts_wsl_optimized_results.json"

if [[ ! -x "$repo_dir/ds4-server" ]]; then
    printf 'Missing executable: %s\n' "$repo_dir/ds4-server" >&2
    exit 2
fi
if [[ ! -r "$model" ]]; then
    printf 'Model is not readable: %s\n' "$model" >&2
    exit 2
fi

mkdir -p "$repo_dir/logs"

env \
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
        --tokens "$max_tokens" \
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
        printf 'Server exited during startup. See %s\n' "$server_log" >&2
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
    printf 'Server did not become ready. See %s\n' "$server_log" >&2
    exit 1
fi

python3 "$script_dir/bench_10_prompts.py" \
    --url "http://127.0.0.1:${port}" \
    --max-tokens "$max_tokens" \
    --output "$result_file"

printf 'Server log: %s\n' "$server_log"

#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(dirname -- "$script_dir")"
name="${1:?usage: run_cuda_experiment_wsl.sh EXPERIMENT_NAME}"
if [[ ! "$name" =~ ^[a-zA-Z0-9._-]+$ ]]; then
    printf 'Invalid experiment name: %s\n' "$name" >&2
    exit 2
fi

model="${DS4_MODEL:-/home/peppe200175/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf}"
port="${DS4_BENCH_PORT:-18080}"
max_tokens="${DS4_BENCH_MAX_TOKENS:-32}"
cache_budget="${DS4_EXPERIMENT_CACHE_BUDGET:-10GB}"
run_id="${DS4_EXPERIMENT_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
run_dir="$repo_dir/logs/cuda_experiments/${run_id}_${name}"
server_log="$run_dir/server.log"
server_stdout="$run_dir/server.stdout.log"
result_file="$run_dir/results.json"
config_file="$run_dir/config.env"
thermal_log="$run_dir/thermal.log"
max_gpu_temp="${DS4_BENCH_MAX_GPU_TEMP:-60}"

if [[ -e "$run_dir" ]]; then
    printf 'Refusing to overwrite experiment directory: %s\n' "$run_dir" >&2
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
    printf 'Another ds4 process is active; refusing a contaminated benchmark.\n' >&2
    pgrep -af 'ds4|ds4-server' >&2 || true
    exit 2
fi

mkdir -p "$run_dir"

gpu_temp=""
for _ in $(seq 1 120); do
    gpu_temp="$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d '[:space:]')"
    printf '%s temp_c=%s\n' "$(date -u +%FT%TZ)" "${gpu_temp:-unavailable}" >>"$thermal_log"
    if [[ "$gpu_temp" =~ ^[0-9]+$ ]] && (( gpu_temp <= max_gpu_temp )); then
        break
    fi
    sleep 5
done
if ! [[ "$gpu_temp" =~ ^[0-9]+$ ]] || (( gpu_temp > max_gpu_temp )); then
    printf 'GPU did not cool below %s C; last value %s C\n' "$max_gpu_temp" "${gpu_temp:-unavailable}" >&2
    exit 2
fi

export DS4_CUDA_MMQ="${DS4_CUDA_MMQ:-0}"
export DS4_CUDA_NO_Q8_F16_CACHE="${DS4_CUDA_NO_Q8_F16_CACHE:-1}"
export DS4_CUDA_STREAMING_READ_THREADS="${DS4_CUDA_STREAMING_READ_THREADS:-4}"
export DS4_CUDA_STREAMING_SMALL_MISS_PARALLEL="${DS4_CUDA_STREAMING_SMALL_MISS_PARALLEL:-1}"
export DS4_CUDA_STREAMING_PERSISTENT_READERS="${DS4_CUDA_STREAMING_PERSISTENT_READERS:-1}"
export DS4_CUDA_DECODE_CACHE_LRU="${DS4_CUDA_DECODE_CACHE_LRU:-1}"
export DS4_CUDA_DECODE_CACHE_EXTRA_EXPERTS="${DS4_CUDA_DECODE_CACHE_EXTRA_EXPERTS:-0}"
export DS4_CUDA_CACHE_SUMMARY="${DS4_CUDA_CACHE_SUMMARY:-1}"
export DS4_CUDA_LAYER_CACHE_CAPACITIES_FILE="${DS4_CUDA_LAYER_CACHE_CAPACITIES_FILE:-0}"
export DS4_CUDA_PINNED_EXPERTS_PREFILL_ONLY="${DS4_CUDA_PINNED_EXPERTS_PREFILL_ONLY:-0}"
export DS4_CUDA_STREAMING_PREFILL_SHARED_OVERLAP="${DS4_CUDA_STREAMING_PREFILL_SHARED_OVERLAP:-0}"
export DS4_CUDA_PROMPT_EXPERT_CACHE="${DS4_CUDA_PROMPT_EXPERT_CACHE:-0}"
export DS4_CUDA_WEIGHT_ARENA_CHUNK_MB="${DS4_CUDA_WEIGHT_ARENA_CHUNK_MB:-256}"
export DS4_LOCK_FILE="/tmp/ds4-cuda-experiment-${run_id}-${name}.lock"

{
    printf 'experiment=%s\n' "$name"
    printf 'run_id=%s\n' "$run_id"
    printf 'git_commit=%s\n' "$(git -C "$repo_dir" rev-parse HEAD)"
    printf 'model=%s\n' "$model"
    printf 'model_size=%s\n' "$(stat -c %s "$model")"
    printf 'max_tokens=%s\n' "$max_tokens"
    printf 'cache_budget=%s\n' "$cache_budget"
    printf 'max_gpu_temp_c=%s\n' "$max_gpu_temp"
    printf 'start_gpu_temp_c=%s\n' "$gpu_temp"
    env | LC_ALL=C sort | grep '^DS4_' || true
} >"$config_file"

"$repo_dir/ds4-server" \
    --cuda \
    -m "$model" \
    --ssd-streaming \
    --ssd-streaming-cache-experts "$cache_budget" \
    --prefill-chunk 128 \
    --ctx 4096 \
    --tokens "$max_tokens" \
    --host 127.0.0.1 \
    --port "$port" \
    >"$server_stdout" 2>"$server_log" &
server_pid=$!
thermal_sampler_pid=""

cleanup() {
    if [[ -n "$thermal_sampler_pid" ]] && kill -0 "$thermal_sampler_pid" 2>/dev/null; then
        kill "$thermal_sampler_pid"
        wait "$thermal_sampler_pid" 2>/dev/null || true
    fi
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

(
    while true; do
        sample="$(nvidia-smi \
            --query-gpu=temperature.gpu,power.draw,pstate,clocks.current.graphics,clocks.current.memory,memory.used \
            --format=csv,noheader,nounits 2>/dev/null | head -n 1)"
        printf '%s sample=%s\n' "$(date -u +%FT%TZ)" "${sample:-unavailable}" >>"$thermal_log"
        sleep 1
    done
) &
thermal_sampler_pid=$!

python3 "$script_dir/bench_10_prompts.py" \
    --url "http://127.0.0.1:${port}" \
    --max-tokens "$max_tokens" \
    --output "$result_file"

kill "$thermal_sampler_pid" 2>/dev/null || true
wait "$thermal_sampler_pid" 2>/dev/null || true
thermal_sampler_pid=""
cleanup
trap - EXIT INT TERM

python3 "$script_dir/summarize_cuda_experiment.py" \
    "$result_file" --server-log "$server_log" --config "$config_file" \
    --output "$run_dir/summary.json"

printf 'Experiment: %s\n' "$run_dir"

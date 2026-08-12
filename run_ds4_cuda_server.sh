#!/usr/bin/env bash
# ds4-server launcher for this machine (RTX 5080, WSL), mirroring the measured
# CLI environment from run_ds4_cuda.sh and the server parameters from
# run-nvidia-tp-server.sh, adapted to the single local GPU.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

if [[ ! -x ./ds4-server ]]; then
    printf 'DS4 launcher: %s/ds4-server is missing or not executable.\n' "$script_dir" >&2
    exit 126
fi

export CUDA_HOME=/home/peppe200175/.local/cuda-13.3.1
export PATH=/home/peppe200175/.local/cuda-13.3.1/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export LD_LIBRARY_PATH=/home/peppe200175/.local/cuda-13.3.1/lib64

# Compact selected-expert IQ2 MMQ is supported, but it did not provide a
# repeatable prefill gain on this RTX 5080, so retain the faster measured tier.
export DS4_CUDA_MMQ="${DS4_CUDA_MMQ:-0}"
export DS4_CUDA_NO_Q8_F16_CACHE=1

# Four direct-I/O workers keep this machine's NVMe queue busy while the
# selected expert spans are uploaded to the RTX 5080 staging/cache buffers.
# Reuse those workers and allow a 3-span single-expert miss to use three lanes.
export DS4_CUDA_STREAMING_READ_THREADS="${DS4_CUDA_STREAMING_READ_THREADS:-4}"
export DS4_CUDA_STREAMING_SMALL_MISS_PARALLEL="${DS4_CUDA_STREAMING_SMALL_MISS_PARALLEL:-1}"
export DS4_CUDA_STREAMING_PERSISTENT_READERS="${DS4_CUDA_STREAMING_PERSISTENT_READERS:-1}"
# On multi-node Linux hosts, pin only these I/O workers to the GPU-local NUMA
# node. The CUDA runtime leaves single-node machines and all compute threads
# untouched.
export DS4_CUDA_STREAMING_NUMA_AFFINITY="${DS4_CUDA_STREAMING_NUMA_AFFINITY:-1}"

# Decode LRU improved cache reuse for this prompt mix. Prefill/shared overlap
# is correct but measured neutral, so leave that experimental path disabled.
export DS4_CUDA_DECODE_CACHE_LRU="${DS4_CUDA_DECODE_CACHE_LRU:-1}"
export DS4_CUDA_DYNAMIC_TIER_PROMOTION="${DS4_CUDA_DYNAMIC_TIER_PROMOTION:-1}"
export DS4_CUDA_STREAMING_PREFILL_SHARED_OVERLAP="${DS4_CUDA_STREAMING_PREFILL_SHARED_OVERLAP:-0}"

# The prompt-aware policy is experimental. A/B tests on this machine did not
# show a repeatable speedup, so keep the measured baseline unless explicitly
# enabled by the caller.
export DS4_CUDA_PROMPT_EXPERT_CACHE="${DS4_CUDA_PROMPT_EXPERT_CACHE:-0}"
export DS4_CUDA_PREFIX_EXPERT_CACHE="${DS4_CUDA_PREFIX_EXPERT_CACHE:-0}"

# The record profile predates the experimental cache/trace paths. Force them
# out of the latency-sensitive launcher; the A/B runner can still enable them.
unset DS4_CUDA_DECODE_CACHE_EXTRA_EXPERTS
unset DS4_CUDA_LAYER_CACHE_CAPACITIES_FILE
unset DS4_CUDA_PINNED_EXPERTS_FILE
unset DS4_CUDA_PINNED_EXPERTS_PREFILL_ONLY
unset DS4_CUDA_CACHE_SUMMARY
unset DS4_EXPERT_TRACE
unset DS4_EXPERT_TRACE_LOGITS

# The upstream weight arena defaults to 1792 MiB chunks. On this 16 GiB GPU,
# smaller chunks avoid a late 1792 MiB reservation after the expert cache and
# most model tensors are resident, while retaining the same tensor-cache size.
export DS4_CUDA_WEIGHT_ARENA_CHUNK_MB="${DS4_CUDA_WEIGHT_ARENA_CHUNK_MB:-256}"
unset DS4_CUDA_MODEL_COPY_CHUNK_MB

# A 128K maximum no longer reserves every compressed KV row up front.  The
# cache starts at a 4K physical allocation and grows losslessly by layer.  The
# saved VRAM supports the measured 8 GB expert cache without changing tokens.
export DS4_CUDA_LAZY_KV_CACHE="${DS4_CUDA_LAZY_KV_CACHE:-1}"
export DS4_CUDA_LAZY_KV_INITIAL_TOKENS="${DS4_CUDA_LAZY_KV_INITIAL_TOKENS:-4096}"

# Server-side knobs (from run-nvidia-tp-server.sh, scaled to this host).
MODEL="${DS4_MODEL:-$script_dir/ds4flash.gguf}"
CTX="${DS4_CTX:-131072}"
SERVER_HOST="${DS4_SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${DS4_SERVER_PORT:-18099}"
KV_DIR="${DS4_KV_DIR:-/tmp/ds4-kv}"
KV_SPACE_MB="${DS4_KV_SPACE_MB:-8192}"

mkdir -p "$KV_DIR"

# Single local GPU: no --cuda-tensor-parallel / --gpu-devices list here (that
# belongs to the 8-GPU TP host).  Engine flags match run_ds4_cuda.sh.
exec ./ds4-server \
    --cuda \
    --model "$MODEL" \
    --ssd-streaming \
    --ssd-streaming-cache-experts 8GB \
    --prefill-chunk 1024 \
    --ctx "$CTX" \
    --host "$SERVER_HOST" \
    --port "$SERVER_PORT" \
    --kv-disk-dir "$KV_DIR" \
    --kv-disk-space-mb "$KV_SPACE_MB" \
    "$@"

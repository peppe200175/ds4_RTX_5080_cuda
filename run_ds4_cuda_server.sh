#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

export CUDA_HOME="${CUDA_HOME:-/home/peppe200175/.local/cuda-13.3.1}"
export PATH="$CUDA_HOME/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64"

# Correctness-tested RTX 5080 profile. The persistent expert cache is filled
# only after the established sequential SSD loader has produced valid data.
export DS4_CUDA_MMQ="${DS4_CUDA_MMQ:-0}"
export DS4_CUDA_NO_Q8_F16_CACHE=1
export DS4_CUDA_WEIGHT_ARENA_CHUNK_MB="${DS4_CUDA_WEIGHT_ARENA_CHUNK_MB:-256}"
export DS4_CUDA_LAZY_KV_CACHE="${DS4_CUDA_LAZY_KV_CACHE:-1}"
export DS4_CUDA_LAZY_KV_INITIAL_TOKENS="${DS4_CUDA_LAZY_KV_INITIAL_TOKENS:-4096}"

# Preserve the proven environment contract for future compatible loaders.
# The verified core currently ignores unsupported knobs rather than enabling
# the old reordered batch-read path that corrupted model output.
export DS4_CUDA_STREAMING_READ_THREADS="${DS4_CUDA_STREAMING_READ_THREADS:-4}"
export DS4_CUDA_STREAMING_SMALL_MISS_PARALLEL="${DS4_CUDA_STREAMING_SMALL_MISS_PARALLEL:-1}"
export DS4_CUDA_STREAMING_PERSISTENT_READERS="${DS4_CUDA_STREAMING_PERSISTENT_READERS:-1}"
export DS4_CUDA_STREAMING_NUMA_AFFINITY="${DS4_CUDA_STREAMING_NUMA_AFFINITY:-1}"
export DS4_CUDA_DECODE_CACHE_LRU="${DS4_CUDA_DECODE_CACHE_LRU:-1}"
export DS4_CUDA_DYNAMIC_TIER_PROMOTION="${DS4_CUDA_DYNAMIC_TIER_PROMOTION:-1}"

exec ./ds4-server \
    --cuda \
    -m "$script_dir/ds4flash.gguf" \
    --ssd-streaming \
    --ssd-streaming-cache-experts "${DS4_SERVER_EXPERT_CACHE:-7GB}" \
    --prefill-chunk "${DS4_SERVER_PREFILL_CHUNK:-1024}" \
    --ctx "${DS4_SERVER_CTX:-135168}" \
    --host "${DS4_SERVER_HOST:-127.0.0.1}" \
    --port "${DS4_SERVER_PORT:-18099}" \
    --cors \
    "$@"

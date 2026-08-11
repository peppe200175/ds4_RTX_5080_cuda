#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

# Always run the executable currently installed in this WSL project. Pulls and
# rebuilds take effect immediately without updating a pinned source/binary hash.
if [[ ! -x ./ds4 ]]; then
    printf 'DS4 launcher: %s/ds4 is missing or not executable.\n' "$script_dir" >&2
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

# Layer-pinned expert profiles remain opt-in: they reduce a few SSD misses but
# did not improve median prefill/decode time on the measured prompt mix.
# Example:
# export DS4_CUDA_PINNED_EXPERTS_FILE="$PWD/profiles/ds4-flash-0731-prefill-pinned-conservative.txt"

# Optional byte-identical copy on a second physical SSD. Reads are assigned in
# deterministic 4 MiB stripes; do not point this at a second path on one SSD.
# export DS4_CUDA_MODEL_REPLICA_PATH=/mnt/second-ssd/ds4flash.gguf

# Server-batch audit: reports one unioned routed dispatch per layer (two owner
# kernels under tensor parallelism), including cumulative grouped row counts.
# export DS4_CUDA_SESSION_BATCH_PROFILE=1
# export DS4_CUDA_SESSION_BATCH_SSD_UNION=1  # measured slower; audit only

# Demand-prefetch effectiveness and async-loader contention. Predictive
# prefetch should only be enabled after this reports low waits/contention.
# export DS4_CUDA_PREFETCH_TELEMETRY=1

exec ./ds4 \
    --cuda \
    -m /home/peppe200175/ds4/ds4flash.gguf \
    --ssd-streaming \
    --ssd-streaming-cache-experts 8GB \
    --prefill-chunk 1024 \
    --ctx 131072 \
    --nothink \
    "$@"

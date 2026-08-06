#!/usr/bin/env bash
set -euo pipefail

cd /mnt/c/Users/gmart/OneDrive/ds4

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

# Decode LRU improved cache reuse for this prompt mix. Prefill/shared overlap
# is correct but measured neutral, so leave that experimental path disabled.
export DS4_CUDA_DECODE_CACHE_LRU="${DS4_CUDA_DECODE_CACHE_LRU:-1}"
export DS4_CUDA_STREAMING_PREFILL_SHARED_OVERLAP="${DS4_CUDA_STREAMING_PREFILL_SHARED_OVERLAP:-0}"

# The prompt-aware policy is experimental. A/B tests on this machine did not
# show a repeatable speedup, so keep the measured baseline unless explicitly
# enabled by the caller.
export DS4_CUDA_PROMPT_EXPERT_CACHE="${DS4_CUDA_PROMPT_EXPERT_CACHE:-0}"

# The record profile predates the experimental cache/trace paths. Force them
# out of the latency-sensitive launcher; the A/B runner can still enable them.
unset DS4_CUDA_DECODE_CACHE_EXTRA_EXPERTS
unset DS4_CUDA_LAYER_CACHE_CAPACITIES_FILE
unset DS4_CUDA_PINNED_EXPERTS_FILE
unset DS4_CUDA_PINNED_EXPERTS_PREFILL_ONLY
unset DS4_CUDA_CACHE_SUMMARY
unset DS4_EXPERT_TRACE_FILE
unset DS4_EXPERT_TRACE_LOGITS

# The upstream weight arena defaults to 1792 MiB chunks. On this 16 GiB GPU,
# smaller chunks avoid a late 1792 MiB reservation after the expert cache and
# most model tensors are resident, while retaining the same tensor-cache size.
export DS4_CUDA_WEIGHT_ARENA_CHUNK_MB="${DS4_CUDA_WEIGHT_ARENA_CHUNK_MB:-256}"
unset DS4_CUDA_MODEL_COPY_CHUNK_MB

# Layer-pinned expert profiles remain opt-in: they reduce a few SSD misses but
# did not improve median prefill/decode time on the measured prompt mix.
# Example:
# export DS4_CUDA_PINNED_EXPERTS_FILE="$PWD/profiles/ds4-flash-0731-prefill-pinned-conservative.txt"

exec ./ds4 \
    --cuda \
    -m /home/peppe200175/ds4/ds4flash.gguf \
    --ssd-streaming \
    --ssd-streaming-cache-experts 10GB \
    --prefill-chunk 128 \
    --ctx 4096 \
    --nothink \
    "$@"

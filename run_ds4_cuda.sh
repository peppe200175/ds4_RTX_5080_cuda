#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

# This launcher is intentionally pinned to the measured definitive CUDA build.
# A pull, rebuild, or accidental switch to another experimental backend must
# not be started under this profile without being verified first.
readonly DS4_RECOVERED_CUDA_BLOB="a7de59b363cc26ce533da7737a99f0b4b714e71f"
readonly DS4_RECOVERED_BINARY_SHA256="789477561f110164da7f0a7377b89193966c4a5a1eb73f60a8b676f9c2514e09"

if ! command -v git >/dev/null 2>&1 || ! command -v sha256sum >/dev/null 2>&1; then
    printf 'DS4 recovered guard: git and sha256sum are required.\n' >&2
    exit 126
fi

# The shared Windows worktree uses CRLF while the preserved Git blob uses LF.
# Normalize only line endings before hashing so Windows Git and WSL Git agree.
actual_cuda_blob="$(tr -d '\r' < ds4_cuda.cu | git hash-object --stdin 2>/dev/null || true)"
actual_binary_sha256="$(sha256sum ./ds4 2>/dev/null | awk '{print $1}')"
if [[ "$actual_cuda_blob" != "$DS4_RECOVERED_CUDA_BLOB" ||
      "$actual_binary_sha256" != "$DS4_RECOVERED_BINARY_SHA256" ]]; then
    printf '%s\n' \
        'DS4 recovered guard: refusing to launch an unapproved build.' \
        "  expected CUDA blob: $DS4_RECOVERED_CUDA_BLOB" \
        "  actual CUDA blob:   ${actual_cuda_blob:-unavailable}" \
        "  expected binary:    $DS4_RECOVERED_BINARY_SHA256" \
        "  actual binary:      ${actual_binary_sha256:-unavailable}" >&2
    exit 126
fi

if [[ "${1:-}" == "--verify-recovered" ]]; then
    printf 'DS4 recovered guard: verified %s\n' "$DS4_RECOVERED_CUDA_BLOB"
    exit 0
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

# A 128K maximum no longer reserves every compressed KV row up front.  The
# cache starts at a 4K physical allocation and grows losslessly by layer.  The
# saved VRAM supports the measured 8 GB expert cache without changing tokens.
export DS4_CUDA_LAZY_KV_CACHE="${DS4_CUDA_LAZY_KV_CACHE:-1}"
export DS4_CUDA_LAZY_KV_INITIAL_TOKENS="${DS4_CUDA_LAZY_KV_INITIAL_TOKENS:-4096}"

# Layer-pinned expert profiles remain opt-in: they reduce a few SSD misses but
# did not improve median prefill/decode time on the measured prompt mix.
# Example:
# export DS4_CUDA_PINNED_EXPERTS_FILE="$PWD/profiles/ds4-flash-0731-prefill-pinned-conservative.txt"

exec ./ds4 \
    --cuda \
    -m /home/peppe200175/ds4/ds4flash.gguf \
    --ssd-streaming \
    --ssd-streaming-cache-experts 8GB \
    --prefill-chunk 1024 \
    --ctx 131072 \
    --nothink \
    "$@"

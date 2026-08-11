#!/usr/bin/env bash
set -euo pipefail

model="${DS4_MODEL:-/models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf}"
if [[ ! -r "$model" ]]; then
    printf 'ds4-run: model not readable: %s\n' "$model" >&2
    printf 'Mount the host GGUF directory at /models or set DS4_MODEL.\n' >&2
    exit 2
fi

exec /opt/ds4/ds4 \
    --cuda \
    -m "$model" \
    --ssd-streaming \
    --ssd-streaming-cache-experts "${DS4_EXPERT_CACHE:-10GB}" \
    --prefill-chunk "${DS4_PREFILL_CHUNK:-128}" \
    --ctx "${DS4_CONTEXT:-4096}" \
    --nothink \
    "$@"

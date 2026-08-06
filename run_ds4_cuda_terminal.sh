#!/usr/bin/env bash
set -uo pipefail

cd /mnt/c/Users/gmart/OneDrive/ds4 || exit 1
mkdir -p logs/interactive
log_file="logs/interactive/ds4_cuda_$(date -u +%Y%m%dT%H%M%SZ).log"
printf 'DS4 CUDA interactive log: %s\n' "$log_file"

./run_ds4_cuda.sh 2> >(tee -a "$log_file" >&2)
status=$?
printf '\nDS4 exited with status %d. Log: %s\n' "$status" "$log_file" >&2
printf 'An interactive WSL shell will remain open for inspection or restart.\n' >&2
exec bash -i

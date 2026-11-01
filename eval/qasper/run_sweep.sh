#!/usr/bin/env bash
# Run the retrieval sweep in parallel shards, then summarise.
# Usage: eval/qasper/run_sweep.sh <results-folder> <shards> <threads-per-shard> [extra options]
set -euo pipefail
out=$1 shards=$2 threads=$3
shift 3
cd "$(dirname "$0")/../.."
echo "== start $(date -Is)"
for ((i = 0; i < shards; i++)); do
    CUDA_VISIBLE_DEVICES= uv run --frozen --group eval python eval/qasper/retrieval_sweep.py run \
        --out "$out" --shard "$i" --shards "$shards" --threads "$threads" "$@" > "$out/shard_$i.log" 2>&1 &
done
wait
uv run --frozen --group eval python eval/qasper/retrieval_sweep.py summarize --out "$out"
echo "== finished $(date -Is)"

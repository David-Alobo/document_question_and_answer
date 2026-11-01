#!/usr/bin/env bash
# Run every stage of the QASPER evaluation; each stage resumes if interrupted.
# Usage: eval/qasper/run_all.sh <results-folder> [ollama-url] [extra run_eval.py options]
set -euo pipefail
out=$1
ollama=${2:-http://127.0.0.1:11434}
shift $(( $# >= 2 ? 2 : 1 ))
cd "$(dirname "$0")/../.."
run() {
    echo "== $* $(date -Is)"
    CUDA_VISIBLE_DEVICES= uv run --frozen python eval/qasper/run_eval.py "$@"
}
for stage in prepare retrieve generate judge; do
    run "$stage" --out "$out" --ollama "$ollama" "$@"
done
# Judgements that came back empty are graded again with a larger budget (appended)
run judge --retry-errors --out "$out" --ollama "$ollama" "$@"
run summarize --out "$out" --ollama "$ollama" "$@"
echo "== finished $(date -Is)"

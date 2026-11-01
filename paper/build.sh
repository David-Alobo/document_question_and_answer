#!/usr/bin/env bash
# Build the paper and SI; fail on LaTeX errors or undefined references. Run from anywhere.
set -euo pipefail
cd "$(dirname "$0")"
for doc in main si; do
    latexmk "$doc.tex" > /dev/null 2>&1 || { tail -30 "build/$doc.log"; exit 1; }
    if grep -E "undefined|Undefined" "build/$doc.log" | grep -v "^Package rerunfilecheck"; then
        echo "$doc: undefined references"; exit 1
    fi
done
grep -c "Overfull" build/main.log build/si.log || true
ls -la build/main.pdf build/si.pdf

#!/usr/bin/env bash
# Fetch the pinned skill repos from corpus/sources.txt into DEST and build the corpus JSONL.
# Usage: scripts/fetch_corpus.sh [DEST] [OUT]   (defaults: .corpus-src corpus.jsonl)
set -euo pipefail
cd "$(dirname "$0")/.."
DEST=${1:-.corpus-src}
OUT=${2:-corpus.jsonl}
mkdir -p "$DEST"
grep -v '^#' corpus/sources.txt | while read -r dir url sha _; do
  [ -n "$dir" ] || continue
  if [ "$(git -C "$DEST/$dir" rev-parse HEAD 2>/dev/null)" != "$sha" ]; then
    rm -rf "${DEST:?}/$dir" && git init -q "$DEST/$dir"
    git -C "$DEST/$dir" fetch -q --depth 1 "$url" "$sha"
    git -C "$DEST/$dir" checkout -q FETCH_HEAD
  fi
  echo "$dir @ ${sha:0:12}"
done
python -m skilljit ingest "$DEST"/* corpus/decoys -o "$OUT"

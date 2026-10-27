#!/usr/bin/env bash
# Parallel ranged download of HotpotQA 2017-10-01 abstracts-with-links dump.
set -euo pipefail
URL=https://nlp.stanford.edu/projects/hotpotqa/enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2
OUT_DIR="$1"; SIZE=1553565403; N=8; CH=$(( (SIZE + N - 1) / N ))
mkdir -p "$OUT_DIR/abstracts_parts"
for i in $(seq 0 $((N-1))); do
  s=$((i*CH)); e=$(( s+CH-1 )); [ $e -ge $SIZE ] && e=$((SIZE-1))
  part="$OUT_DIR/abstracts_parts/part$i"; want=$((e-s+1))
  ( for t in 1 2 3 4 5 6 7 8; do
      have=$( [ -f "$part" ] && stat -c %s "$part" || echo 0 )
      [ "$have" -ge "$want" ] && break
      curl -sL --retry 5 -r $((s+have))-$e "$URL" >> "$part" || sleep 5
    done ) &
done
wait
for i in $(seq 0 $((N-1))); do cat "$OUT_DIR/abstracts_parts/part$i"; done > "$OUT_DIR/abstracts.tar.bz2"
[ "$(stat -c %s "$OUT_DIR/abstracts.tar.bz2")" -eq $SIZE ] && rm -r "$OUT_DIR/abstracts_parts" && echo "abstracts OK"

#!/usr/bin/env bash
# Resilient download: fetch URL in fixed-size byte ranges with P parallel workers,
# retrying each range until complete, and write the ranges to stdout in order.
# Only a few chunks live on disk at a time, so this can feed a decompressor directly.
# Usage: fetch_ranges.sh URL TMP_DIR [CHUNK_MB] [PARALLEL] [START_BYTE] > stream
set -uo pipefail
URL=$1; TMP=$2; CHUNK=$(( ${3:-256} * 1024 * 1024 )); P=${4:-4}; START=${5:-0}
mkdir -p "$TMP"
SIZE=$(curl -sIL "$URL" | grep -i '^content-length' | tail -1 | tr -dc 0-9)
[ -n "$SIZE" ] || { echo "cannot get size" >&2; exit 1; }
N=$(( (SIZE - START + CHUNK - 1) / CHUNK ))
echo "size $SIZE, $N chunks" >&2

fetch() {  # fetch chunk i into $TMP/i, mark $TMP/i.done
  local i=$1 s=$(( START + $1 * CHUNK )) e want have tries=0
  e=$(( s + CHUNK - 1 )); [ $e -ge $SIZE ] && e=$(( SIZE - 1 )); want=$(( e - s + 1 ))
  while :; do
    have=$( [ -f "$TMP/$i" ] && stat -c %s "$TMP/$i" || echo 0 )
    [ "$have" -ge "$want" ] && break
    curl -sL --max-time 900 -r $(( s + have ))-$e "$URL" >> "$TMP/$i" || { tries=$((tries+1)); echo "chunk $i retry $tries" >&2; sleep $(( tries < 10 ? tries * 3 : 30 )); }
  done
  touch "$TMP/$i.done"
}

next=0
for (( i = 0; i < N; i++ )); do
  # keep up to P chunks in flight ahead of the consumer
  while (( next < N && next < i + P )); do fetch $next & next=$((next+1)); done
  while [ ! -f "$TMP/$i.done" ]; do sleep 1; done
  cat "$TMP/$i" && rm -f "$TMP/$i" "$TMP/$i.done"
  echo "chunk $((i+1))/$N" >&2
done
wait

#!/usr/bin/env bash
set -euo pipefail

wheel=/root/autodl-tmp/kfair/cache/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
url=https://ghproxy.net/https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
total=187814472
workers=8
start=$(stat -c %s "$wheel")
remaining=$((total - start))
chunk=$(((remaining + workers - 1) / workers))

if (( start <= 0 || start >= total )); then
  echo "unexpected starting size: $start" >&2
  exit 1
fi

pids=()
parts=()
ends=()
for ((i=0; i<workers; i++)); do
  range_start=$((start + i * chunk))
  (( range_start >= total )) && break
  range_end=$((range_start + chunk - 1))
  (( range_end >= total )) && range_end=$((total - 1))
  part="${wheel}.part.${i}"
  parts+=("$part")
  ends+=("$((range_end - range_start + 1))")
  curl --http1.1 -L --fail --retry 5 --retry-all-errors \
    --range "${range_start}-${range_end}" -o "$part" "$url" &
  pids+=("$!")
done

for pid in "${pids[@]}"; do wait "$pid"; done
for ((i=0; i<${#parts[@]}; i++)); do
  actual=$(stat -c %s "${parts[$i]}")
  expected=${ends[$i]}
  if (( actual != expected )); then
    echo "part $i size mismatch: $actual != $expected" >&2
    exit 1
  fi
done

for part in "${parts[@]}"; do
  dd if="$part" of="$wheel" oflag=append conv=notrunc status=none
done

final=$(stat -c %s "$wheel")
if (( final != total )); then
  echo "final size mismatch: $final != $total" >&2
  exit 1
fi

rm -f -- "${parts[@]}"
echo "PARALLEL_DOWNLOAD_READY size=$final"

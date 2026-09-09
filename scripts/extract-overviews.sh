#!/usr/bin/env bash
# Extract slide overviews for the tissue-detection check.
#
# Tissue detection is judged on real slides or not at all: the failure modes
# that matter — a faded section missed, two fragments welded into one — do not
# appear on synthetic images. This pulls a thumbnail out of every slide it is
# pointed at, using libvips (which reads SVS/NDPI/MRXS through OpenSlide), so
# the check can run headlessly over a whole set.
#
#   scripts/extract-overviews.sh ~/slides ~/more/slides
#
# Then: npm run verify:tissue
set -euo pipefail

OUT="${SLIDECRAFT_OVERVIEWS:-/tmp/slidecraft-overviews}"
mkdir -p "$OUT"

if ! command -v vips >/dev/null; then
  echo "libvips not found. brew install vips" >&2
  exit 1
fi

count=0
for dir in "$@"; do
  while IFS= read -r f; do
    name=$(basename "$f"); name="${name%.*}"
    if vips thumbnail "$f" "$OUT/$name.ppm" 1400 2>/dev/null; then
      vips copy "$OUT/$name.ppm" "$OUT/$name.png" 2>/dev/null || true
      count=$((count + 1))
      printf '.'
    else
      printf '!'
    fi
  done < <(find "$dir" -maxdepth 3 \( -name '*.svs' -o -name '*.ndpi' -o -name '*.mrxs' -o -name '*.tif' -o -name '*.tiff' \))
done
printf '\n%d overviews in %s\n' "$count" "$OUT"

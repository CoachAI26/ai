#!/usr/bin/env bash
set -euo pipefail

# Usage:
# ./scripts/transcribe_pretty.sh /path/to/audio.wav [server_url] [level] [category] [title] [out_file]

FILE="${1:-/Users/khaneapple/seon_ai/5825784783795408424.wav}"
URL="${2:-http://localhost:8000}"
LEVEL="${3:-B1}"
CATEGORY="${4:-Speaking}"
TITLE="${5:-My Presentation}"
OUT="${6:-resp_$(date +%Y%m%dT%H%M%S).json}"

if [ ! -f "$FILE" ]; then
  echo "File not found: $FILE" >&2
  exit 2
fi

# detect mime type from extension
ext="${FILE##*.}"
# lowercase extension in a macOS-compatible way (bash 3.2 doesn't support ${var,,})
ext_lc="$(echo "$ext" | tr '[:upper:]' '[:lower:]')"
case "$ext_lc" in
  wav) MIME="audio/wav";;
  mp3) MIME="audio/mpeg";;
  m4a) MIME="audio/x-m4a";;
  mp4) MIME="audio/mp4";;
  ogg) MIME="audio/ogg";;
  flac) MIME="audio/flac";;
  webm) MIME="audio/webm";;
  aac) MIME="audio/x-aac";;
  *) MIME="application/octet-stream";;
esac

echo "Posting $FILE to $URL/api/v1/transcribe (mime=$MIME)"

curl -s -X POST "$URL/api/v1/transcribe" \
  -H "accept: application/json" \
  -F "file=@${FILE};type=${MIME}" \
  -F "level=${LEVEL}" \
  -F "category=${CATEGORY}" \
  -F "title=${TITLE}" \
  | jq '.' > "$OUT"

if [ $? -eq 0 ]; then
  echo "Saved pretty JSON to $OUT"
  echo "----- Preview -----"
  jq . "$OUT" | sed -n '1,200p'
else
  echo "Request failed or jq not installed." >&2
  exit 1
fi

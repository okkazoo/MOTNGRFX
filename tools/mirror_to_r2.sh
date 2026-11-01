#!/bin/bash
# Mirror the models for one VRAM tier from their origin (Hugging Face) to R2.
# Source of truth: models.json at the repo root. R2 key == ComfyUI-relative dest
# (e.g. models/text_encoders/x.safetensors), same convention as the card-library
# _mirror_to_r2.sh scripts. Bucket r2:model-loras, public base below.
#
# RUN THIS ON A FAST-NETWORK MACHINE (a Vast instance), not a home PC:
# `rclone copyurl` streams every byte through the host running it (HF -> host -> R2).
#
#   bash tools/mirror_to_r2.sh --tier 24            # mirror tier 24 files
#   bash tools/mirror_to_r2.sh --tier 16 --dry-run  # print the plan only
#
# Behaviour:
#   - one read-only `rclone lsjson` listing of r2:model-loras/models up front;
#     keys already on R2 with the manifest size are skipped
#   - gated origin files (Lightricks/LTX-2.5) need HF_TOKEN from an account that
#     accepted the license: the HF URL is resolved to its signed CDN URL with the
#     token (passed to curl on stdin, never on argv), then copyurl'd
#   - after copying, every key for the tier is HEAD-verified on the public URL
#     (Content-Length == manifest size_bytes)
#   - --dry-run makes no writes: it lists R2 (read-only) and prints what it would do
# Mirror all tiers by running --tier 16, 24 and 32 (16 has the GGUF, 32 has MOSS bf16).
# NOTE: explicit dest path (no -a), --s3-no-check-bucket (token can't CreateBucket).
set -u

R2="r2:model-loras"
PUBLIC_BASE="https://pub-ad01ab5c49c349979d7ee68a67eb34c9.r2.dev"
RC="rclone copyurl --s3-no-check-bucket --retries 5 --low-level-retries 10"

TIER=24
DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --tier) TIER="${2:-}"; shift 2 ;;
    --tier=*) TIER="${1#--tier=}"; shift ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) sed -n '2,27p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
case "$TIER" in 12|16|24|32) ;; *) echo "--tier must be 12, 16, 24 or 32" >&2; exit 2 ;; esac

cd "$(dirname "$0")/.." || exit 1
PY=""
for c in python3 python; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import json' >/dev/null 2>&1; then PY="$c"; break; fi
done
[ -n "$PY" ] || { echo "python not found" >&2; exit 1; }

# url|key|size|gated  for every manifest entry in this tier
mapfile -t ROWS < <("$PY" -c '
import json, sys
tier = int(sys.argv[1])
for m in json.load(open("models.json", encoding="utf-8"))["models"]:
    if tier in m["tiers"]:
        print("%s|%s|%d|%d" % (m["origin_url"], m["r2_key"], m["size_bytes"], 1 if m["gated"] else 0))
' "$TIER" | tr -d '\r')
[ "${#ROWS[@]}" -gt 0 ] || { echo "no models for tier $TIER" >&2; exit 1; }

human() { "$PY" -c 'import sys; print("%.2f GB" % (int(sys.argv[1]) / 1024**3))' "$1" | tr -d '\r'; }

# Existing R2 objects: key|size (read-only listing)
declare -A R2SIZE=()
R2_OK=0
if command -v rclone >/dev/null 2>&1; then
  LISTING=$(rclone lsjson -R --files-only --no-modtime --no-mimetype "$R2/models" 2>/dev/null) && R2_OK=1
  if [ "$R2_OK" = 1 ]; then
    while IFS='|' read -r k s; do
      [ -n "$k" ] && R2SIZE["$k"]="$s"
    done < <(printf '%s' "$LISTING" | "$PY" -c '
import json, sys
for o in json.load(sys.stdin) or []:
    print("models/%s|%d" % (o["Path"], o["Size"]))
' | tr -d '\r')
  fi
fi
if [ "$R2_OK" != 1 ]; then
  echo "WARNING: could not list $R2/models (rclone missing or remote not configured); assuming nothing is on R2."
fi

total=0; for row in "${ROWS[@]}"; do IFS='|' read -r _ _ size _ <<<"$row"; total=$((total + size)); done
echo "=== Tier $TIER: ${#ROWS[@]} files, $(human $total)  ->  $R2   (dry-run=$DRY)"

ok=0; skip=0; fail=0; todo_bytes=0
for row in "${ROWS[@]}"; do
  IFS='|' read -r url key size gated <<<"$row"
  have="${R2SIZE[$key]:-}"
  if [ "$have" = "$size" ]; then
    echo "SKIP (on R2, $size bytes)  $key"; skip=$((skip+1)); continue
  fi
  [ -n "$have" ] && echo "    R2 has $key with $have bytes, expected $size: will overwrite"
  todo_bytes=$((todo_bytes + size))
  if [ "$DRY" = 1 ]; then
    note=""; [ "$gated" = 1 ] && note="  [gated: needs HF_TOKEN]"
    echo "WOULD: $RC \"$url\" \"$R2/$key\"$note"
    continue
  fi

  src="$url"
  if [ "$gated" = 1 ]; then
    if [ -z "${HF_TOKEN:-}" ]; then
      echo "FAIL: $key is gated and HF_TOKEN is not set"; fail=$((fail+1)); continue
    fi
    src=$(printf 'header = "Authorization: Bearer %s"\n' "$HF_TOKEN" \
          | curl -sS -K - -o /dev/null -w '%{redirect_url}' "$url")
    if [ -z "$src" ]; then
      echo "FAIL: $key: no signed redirect from HF (license not accepted / bad token?)"; fail=$((fail+1)); continue
    fi
  fi

  echo "=== $(date '+%H:%M:%S')  ->  $key  ($(human "$size"))"
  if $RC "$src" "$R2/$key"; then
    echo "    OK: $key"; ok=$((ok+1))
  else
    echo "    FAIL: $key"; fail=$((fail+1))
  fi
done
echo "=== MIRROR $([ "$DRY" = 1 ] && echo PLAN || echo DONE): $ok copied, $skip already on R2, $fail failed; to copy: $(human $todo_bytes)"

if [ "$DRY" = 1 ]; then
  exit 0
fi

echo "=== HEAD-verifying public URLs"
bad=0
for row in "${ROWS[@]}"; do
  IFS='|' read -r _ key size _ <<<"$row"
  len=$(curl -sI "$PUBLIC_BASE/$key" | tr -d '\r' | awk 'tolower($1)=="content-length:"{print $2}' | tail -1)
  if [ "$len" = "$size" ]; then
    echo "  200 $size  $key"
  else
    echo "  BAD (got '${len:-none}', expected $size)  $key"; bad=$((bad+1))
  fi
done
echo "=== VERIFY: $(( ${#ROWS[@]} - bad )) ok, $bad bad"
[ "$fail" = 0 ] && [ "$bad" = 0 ]

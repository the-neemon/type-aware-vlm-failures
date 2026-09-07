#!/usr/bin/env bash
# Download and unpack ChartQA. Bash port of scripts/download_chartqa.ps1
# (open decision 11): PowerShell does not run on Ada, and the images have to
# land there before inference can start.
#
#   scripts/download_chartqa.sh                      # JSON + images, into $HOME/data/ChartQA
#   scripts/download_chartqa.sh --json-only          # JSON only, for split stats on a laptop
#   scripts/download_chartqa.sh --out /some/where
#
# Produces the layout src/eval/chartqa.py expects:
#
#   <out>/train/train_human.json
#   <out>/train/train_augmented.json
#   <out>/val/val_human.json          ... and so on for test
#   <out>/{train,val,test}/png/*.png  (unless --json-only)
#
# Two deliberate choices.
#
# Output defaults to $HOME, not /scratch. The images are ~850 MB, and on Ada
# /scratch is node-local and purged at 7 days (results/ada_filesystem.md), so
# putting them there means re-downloading on whichever node the scheduler
# picks. $HOME is visible from every node and had ~6 GB free at the time of
# writing, which fits. The zip itself is staged on /scratch where available so
# that the 875 MB never counts against the home quota.
#
# It is idempotent. Re-running with the data already present does nothing, so
# it is safe to call from a job script.
set -euo pipefail

ZIP_URL="https://huggingface.co/datasets/ahmed-masry/ChartQA/resolve/main/ChartQA%20Dataset.zip"
OUT="$HOME/data/ChartQA"
JSON_ONLY=0
KEEP_ZIP=0
WORKDIR=""

usage() { sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

while [ $# -gt 0 ]; do
    case "$1" in
        --out)       OUT="$2"; shift 2 ;;
        --workdir)   WORKDIR="$2"; shift 2 ;;
        --json-only) JSON_ONLY=1; shift ;;
        --keep-zip)  KEEP_ZIP=1; shift ;;
        -h|--help)   usage 0 ;;
        *) echo "unknown argument: $1" >&2; usage 1 ;;
    esac
done

for tool in curl unzip python3; do
    command -v "$tool" >/dev/null 2>&1 || {
        echo "FATAL: '$tool' not found on $(hostname)." >&2; exit 1; }
done

# Stage the zip off the home quota when a big local disk exists.
if [ -z "$WORKDIR" ]; then
    if [ -d /scratch ] && [ -w /scratch ]; then
        WORKDIR="/scratch/vlm-failures/downloads"
    else
        WORKDIR="$OUT/.work"
    fi
fi
mkdir -p "$OUT" "$WORKDIR"
ZIP="$WORKDIR/ChartQA_Dataset.zip"

echo "=== ChartQA download ==="
echo "node    : $(hostname)"
echo "out     : $OUT"
echo "workdir : $WORKDIR"
echo "mode    : $([ "$JSON_ONLY" -eq 1 ] && echo 'JSON only' || echo 'JSON + images')"

# --- already done? ------------------------------------------------------
expected_json=(train/train_human.json train/train_augmented.json
               val/val_human.json     val/val_augmented.json
               test/test_human.json   test/test_augmented.json)
have_all=1
for rel in "${expected_json[@]}"; do
    [ -s "$OUT/$rel" ] || { have_all=0; break; }
done
if [ "$have_all" -eq 1 ] && [ "$JSON_ONLY" -eq 1 ]; then
    echo "All six JSON files already present. Nothing to do."
    exit 0
fi
if [ "$have_all" -eq 1 ] && [ "$JSON_ONLY" -eq 0 ] \
   && [ -n "$(find "$OUT" -name '*.png' -print -quit 2>/dev/null)" ]; then
    echo "JSON and images already present. Nothing to do."
    exit 0
fi

# --- download (resumable) -----------------------------------------------
echo
# unzip -t is the integrity check the PowerShell version never had; a truncated
# download otherwise surfaces later as a confusing JSON parse error. Testing
# first also makes re-runs cheap: a complete archive skips the download, which
# matters because curl -C - on an already-complete file errors rather than
# succeeding.
if [ -s "$ZIP" ] && unzip -tqq "$ZIP" >/dev/null 2>&1; then
    echo "archive already downloaded and intact, skipping download"
else
    echo "downloading ~875 MB (resumable; re-running continues a partial file) ..."
    curl -L --fail --retry 5 --retry-delay 5 -C - -o "$ZIP" "$ZIP_URL"
    echo "verifying archive ..."
    unzip -tqq "$ZIP" >/dev/null 2>&1 || {
        echo "FATAL: archive is corrupt. Delete $ZIP and re-run." >&2; exit 1; }
fi

# --- extract ------------------------------------------------------------
# Discover the archive's top-level prefix rather than hardcoding
# "ChartQA Dataset/", so a re-packaged upstream does not break this silently.
# awk rather than `head -1`: head closes the pipe as soon as it has its line,
# unzip takes SIGPIPE, and under `set -o pipefail` that kills the script with
# exit 141. awk reads the listing to EOF and prints only the first field.
PREFIX="$(unzip -Z1 "$ZIP" | awk -F/ 'NR==1{print $1}')"
[ -n "$PREFIX" ] || { echo "FATAL: could not read the archive listing." >&2; exit 1; }
echo "archive top-level directory: '$PREFIX'"

echo "extracting ..."
TMP="$WORKDIR/extract.$$"
rm -rf "$TMP"; mkdir -p "$TMP"
if [ "$JSON_ONLY" -eq 1 ]; then
    # Mirrors the PowerShell behaviour: JSON, excluding the annotation files.
    unzip -qq "$ZIP" "$PREFIX/*.json" -x "$PREFIX/*annotation*" -d "$TMP"
else
    unzip -qq "$ZIP" -x "$PREFIX/*annotation*" -d "$TMP"
fi

# Strip the top-level prefix, then merge into $OUT.
if [ -d "$TMP/$PREFIX" ]; then
    cp -a "$TMP/$PREFIX/." "$OUT/"
else
    cp -a "$TMP/." "$OUT/"
fi
rm -rf "$TMP"
[ "$KEEP_ZIP" -eq 1 ] || { rm -f "$ZIP"; echo "removed the zip; pass --keep-zip to keep it"; }

# --- verify -------------------------------------------------------------
echo
echo "=== verifying ==="
missing=0
for rel in "${expected_json[@]}"; do
    if [ -s "$OUT/$rel" ]; then
        printf "  %-34s %s\n" "$rel" "$(python3 -c "
import json,sys
rows=json.load(open(sys.argv[1]))
figs={r.get('imgname') for r in rows}
print(f'{len(rows)} questions over {len(figs)} figures')" "$OUT/$rel")"
    else
        printf "  %-34s MISSING\n" "$rel"; missing=1
    fi
done

if [ "$JSON_ONLY" -eq 0 ]; then
    n_png=$(find "$OUT" -name '*.png' | wc -l)
    echo "  images: $n_png png files"
    [ "$n_png" -gt 0 ] || { echo "  FATAL: no images extracted." >&2; missing=1; }
fi

echo "  total on disk: $(du -sh "$OUT" | cut -f1)"
if [ "$missing" -ne 0 ]; then
    echo "FAILED: the layout is incomplete; src/eval/chartqa.py will not load it." >&2
    exit 1
fi
echo "OK. Load it with: python3 -m src.eval.chartqa $OUT"

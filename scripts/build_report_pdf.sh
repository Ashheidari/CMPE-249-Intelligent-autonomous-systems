#!/usr/bin/env bash
# Render report/submission/technical_report.html to PDF with headless Google Chrome.
# Usage: AUTHOR="Your Name" scripts/build_report_pdf.sh   -> report/submission/technical_report.pdf
set -euo pipefail
cd "$(dirname "$0")/.."
SRC=report/submission/technical_report.html
TMP=report/submission/.render.html
OUT=report/submission/technical_report.pdf
CHROME="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
sed "s|{{AUTHOR}}|${AUTHOR:-[Author name]}|" "$SRC" > "$TMP"
"$CHROME" --headless=new --disable-gpu --no-pdf-header-footer --run-all-compositor-stages-before-draw \
  --virtual-time-budget=5000 --print-to-pdf="$OUT" "file://$PWD/$TMP" 2>/dev/null
rm -f "$TMP"
echo "wrote $OUT"

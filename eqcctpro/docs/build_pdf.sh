#!/usr/bin/env bash
# Build EQCCTPro_Draft.pdf from EQCCTPro_Draft.md
# Usage: bash docs/build_pdf.sh
# Run from the eqcctpro/ project root.

set -e

PANDOC=/home/skevofilaxc/miniconda3/envs/eqcctpro/bin/pandoc
TECTONIC=/home/skevofilaxc/miniconda3/bin/tectonic
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$SCRIPT_DIR/EQCCTPro_Draft.md"
OUT="$SCRIPT_DIR/EQCCTPro_Draft.pdf"
HEADER="$SCRIPT_DIR/pdf_header.tex"
TMP=/tmp/EQCCTPro_Draft_abs.md

echo "Pre-processing image paths to absolute..."
python3 - "$SRC" "$SCRIPT_DIR" "$TMP" << 'PY'
import re, sys
from pathlib import Path
src_text = Path(sys.argv[1]).read_text()
abs_dir  = Path(sys.argv[2]).resolve()
def sub(m):
    alt, rel = m.group(1), m.group(2)
    if rel.startswith('/') or rel.startswith('http'):
        return m.group(0)
    return f'![{alt}]({abs_dir}/{rel})'
out = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', sub, src_text)
Path(sys.argv[3]).write_text(out)
PY

echo "Running pandoc -> tectonic..."
"$PANDOC" "$TMP" \
  --pdf-engine="$TECTONIC" \
  --include-in-header="$HEADER" \
  -V geometry:margin=1in \
  -V fontsize=11pt \
  -V colorlinks=true \
  -V linkcolor=blue \
  -o "$OUT"

echo "Done: $OUT ($(du -h "$OUT" | cut -f1))"

#!/bin/bash
# Setup fairseq for Bottleneck_LC: copy from RLFW-NAT and overlay our criterion.
set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FAIRSEQ_SRC="${FAIRSEQ_SRC:-}"
if [ -z "$FAIRSEQ_SRC" ]; then
  for d in "$REPO_ROOT/../RLFW-NAT/fairseq_mask" "$REPO_ROOT/../RLFW-NAT.mono/fairseq_mask" "/Users/alphadl/git/RLFW-NAT/fairseq_mask"; do
    if [ -d "$d" ] && [ -f "$d/fairseq/criterions/nat_loss.py" ]; then
      FAIRSEQ_SRC="$d"
      break
    fi
  done
fi
if [ -z "$FAIRSEQ_SRC" ] || [ ! -d "$FAIRSEQ_SRC" ]; then
  echo "Error: Set FAIRSEQ_SRC to a fairseq tree that contains translation_lev + nat_loss (e.g. RLFW-NAT/fairseq_mask)"
  exit 1
fi
echo "Using fairseq from: $FAIRSEQ_SRC"
FAIRSEQ_DST="$REPO_ROOT/fairseq"
rm -rf "$FAIRSEQ_DST"
cp -R "$FAIRSEQ_SRC" "$FAIRSEQ_DST"
cp "$REPO_ROOT/fairseq_mods/criterions/bottleneck_lc_loss.py" "$FAIRSEQ_DST/fairseq/criterions/"
if grep -q 'l\[\["loss"\]\]' "$FAIRSEQ_DST/fairseq/criterions/nat_loss.py" 2>/dev/null; then
  sed -i.bak 's/l\[\["loss"\]\]/l["loss"]/g' "$FAIRSEQ_DST/fairseq/criterions/nat_loss.py" || true
fi
echo "Installing fairseq in editable mode..."
pip install -e "$FAIRSEQ_DST" --quiet
echo "Done. fairseq is at $FAIRSEQ_DST"

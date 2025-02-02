#!/bin/bash
# Generate translations and compute BLEU. Usage: SRC=en TGT=de DATA=... SAVE=... TEST_PREF=... ./scripts/eval_bleu.sh
set -e
SRC="${SRC:-en}"
TGT="${TGT:-de}"
DATA="${DATA:-./data/${SRC}${TGT}/databin/forward_KD}"
SAVE="${SAVE:-./checkpoints/${SRC}${TGT}/baseline_mask}"
TEST_PREF="${TEST_PREF:-./data/${SRC}${TGT}/test}"
CKPT="${CKPT:-checkpoint_best.pt}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GEN="$SAVE/gen"
mkdir -p "$GEN"
python "$ROOT/fairseq/fairseq_cli/generate.py" "$DATA" \
  --path "$SAVE/$CKPT" -s $SRC -t $TGT \
  --gen-subset test --beam 5 --max-tokens 8192 \
  --batch-size 128 > "$GEN/test.out" 2>"$GEN/test.log"
grep ^H "$GEN/test.out" | cut -f3- > "$GEN/test.hyp"
if [ -f "${TEST_PREF}.${TGT}" ]; then
  sacrebleu "${TEST_PREF}.${TGT}" -i "$GEN/test.hyp" -b
fi

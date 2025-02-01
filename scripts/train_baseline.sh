#!/bin/bash
# Baseline: Mask-Predict or Levenshtein on forward KD only (no prior, no raw pretrain).
set -e
SRC="${SRC:-en}"
TGT="${TGT:-de}"
DATA="${DATA:-./data/${SRC}${TGT}/databin/forward_KD}"
SAVE="${SAVE:-./checkpoints/${SRC}${TGT}/baseline_mask}"
ARCH="${ARCH:-cmlm_transformer}"
TASK="${TASK:-translation_lev}"
CRIT="${CRIT:-nat_loss}"
NOISE="${NOISE:-random_mask}"
[ "$ARCH" = "levenshtein_transformer" ] && NOISE="random_delete"
MAX_UPDATE="${MAX_UPDATE:-300000}"
WARMUP="${WARMUP:-10000}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$SAVE"
python "$ROOT/fairseq/fairseq_cli/train.py" "$DATA" \
  --save-dir "$SAVE" -s $SRC -t $TGT \
  --task $TASK --criterion $CRIT --arch $ARCH \
  --label-smoothing 0.1 --dropout 0.2 --noise $NOISE \
  --share-decoder-input-output-embed --decoder-learned-pos --encoder-learned-pos --apply-bert-init \
  --optimizer adam --adam-betas '(0.9,0.98)' \
  --lr 1e-7 --max-lr 1e-3 --lr-scheduler cosine \
  --warmup-init-lr 1e-7 --warmup-updates $WARMUP --lr-period-updates $((MAX_UPDATE - WARMUP)) \
  --max-update $MAX_UPDATE --weight-decay 0.01 --clip-norm 0.1 \
  --max-tokens 8192 --update-freq 4 --fp16 \
  --save-interval-updates 2000 --keep-last-epochs 5 --seed 1
